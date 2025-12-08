import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import requests
import schedule
from bs4 import BeautifulSoup

# Ensure project root is on sys.path so imports work regardless of CWD
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils.logger import setup_logger


# Configuration (override via env vars)
BASE_URL = os.getenv("JOURNAL_BASE_URL", "https://search.ipindia.gov.in/IPOJournal/Journal/Trademark")
DOWNLOAD_DIR = Path(os.getenv("JOURNAL_DOWNLOAD_DIR", "./data/journal_downloads"))
MAX_WORKERS = int(os.getenv("JOURNAL_MAX_WORKERS", "4"))  # Reserved for future parallelism
SCHEDULE_TIME = os.getenv("JOURNAL_SCHEDULE_TIME", "10:30")  # 24h format
PROCESS_ENDPOINT = os.getenv(
    "PROCESS_PDF_ENDPOINT",
    f"http://{Config.API_HOST}:{Config.API_PORT}/process-pdf-and-index"
)
VERIFY_SSL = os.getenv("JOURNAL_VERIFY_SSL", "false").lower() == "true"
REQUEST_TIMEOUT = int(os.getenv("JOURNAL_REQUEST_TIMEOUT_SECONDS", "60"))
# Processing can take 7–14 minutes; allow a higher timeout for the API call
PROCESS_TIMEOUT = int(os.getenv("JOURNAL_PROCESS_TIMEOUT_SECONDS", "1200"))
PROCESSING_PAUSE_SECONDS = int(os.getenv("JOURNAL_PROCESSING_PAUSE_SECONDS", "120"))
LOCK_FILE = Path(os.getenv("JOURNAL_LOCK_FILE", DOWNLOAD_DIR / ".journal_downloader.lock"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://search.ipindia.gov.in",
    "Referer": "https://search.ipindia.gov.in/IPOJournal/Journal/Trademark",
}

# Ensure log directory exists before logger creation
Config.LOG_PATH.mkdir(parents=True, exist_ok=True)

logger: logging.Logger = setup_logger(
    "journal_downloader",
    Config.LOG_PATH / f"journal_downloader_{datetime.now().strftime('%Y%m%d')}.log",
    level=getattr(logging, Config.LOG_LEVEL),
)


def fetch_download_tasks() -> List[Dict]:
    """Scrape the journal page and return download task descriptors."""
    logger.info("Fetching journal listing page")
    response = requests.get(BASE_URL, headers=HEADERS, verify=VERIFY_SSL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error("Download table not found on page")
        return []

    rows = [row for row in table.find_all("tr") if row.find_all("td")]
    if not rows:
        logger.error("No data rows found in journal table")
        return []

    # Target the first data row (Sr No 1)
    target_row = rows[0]
    cols = target_row.find_all("td")
    if len(cols) < 2:
        logger.error("Unexpected column structure in journal row")
        return []

    journal_no = cols[1].get_text(strip=True) or "unknown"
    download_col = cols[-1]
    forms = download_col.find_all("form")

    download_tasks = []
    for form in forms:
        action = form.get("action")
        input_tag = form.find("input", {"name": "FileName"})
        button = form.find("button")

        if not action or not input_tag:
            continue

        post_url = requests.compat.urljoin(BASE_URL, action)
        file_id = input_tag.get("value")
        raw_name = button.get_text(strip=True) if button else "Unknown"
        safe_name = "".join(ch for ch in raw_name if ch.isalnum() or ch in " -_").strip() or "Unknown"
        filename = f"Journal_{journal_no}_{safe_name}.pdf"
        file_path = DOWNLOAD_DIR / filename

        download_tasks.append(
            {
                "post_url": post_url,
                "payload": {"FileName": file_id},
                "file_path": file_path,
                "file_name": filename,
            }
        )

    logger.info("Prepared %s download task(s) for journal %s", len(download_tasks), journal_no)
    return download_tasks


def download_file(post_url: str, payload: Dict, file_path: Path, file_name: str) -> bool:
    """Stream a single PDF to disk."""
    logger.info("Starting download: %s", file_name)
    try:
        with requests.post(
            post_url,
            data=payload,
            headers=HEADERS,
            verify=VERIFY_SSL,
            stream=True,
            timeout=REQUEST_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("Finished download: %s", file_name)
        return True
    except Exception as exc:
        logger.error("Download failed for %s: %s", file_name, exc)
        return False


def process_pdf(file_path: Path) -> bool:
    """Send the downloaded PDF to the API for processing."""
    logger.info("Sending for processing: %s", file_path.name)
    try:
        with open(file_path, "rb") as pdf_file:
            resp = requests.post(
                PROCESS_ENDPOINT,
                files={"file": (file_path.name, pdf_file, "application/pdf")},
                timeout=PROCESS_TIMEOUT,
                verify=VERIFY_SSL,
            )
        if 200 <= resp.status_code < 300:
            logger.info("Processing succeeded: %s", file_path.name)
            return True
        logger.error("Processing failed (%s) for %s: %s", resp.status_code, file_path.name, resp.text)
        return False
    except Exception as exc:
        logger.error("Processing error for %s: %s", file_path.name, exc)
        return False


def run_cycle():
    """Download the latest journal PDFs and process them sequentially."""
    # Prevent overlapping runs (e.g., if schedule triggers while a run is still active)
    lock_fd = None
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        logger.info("Another run is already in progress; skipping this cycle.")
        return
    except Exception as exc:
        logger.error("Failed to acquire lock: %s", exc)
        return

    logger.info("--- Journal task started %s ---", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tasks = fetch_download_tasks()
    if not tasks:
        logger.info("No tasks found; nothing to download.")
        return _release_lock(lock_fd)

    for task in tasks:
        file_path: Path = task["file_path"]
        if file_path.exists():
            logger.info("Already downloaded, skipping: %s", file_path.name)
            continue

        downloaded = download_file(
            post_url=task["post_url"],
            payload=task["payload"],
            file_path=file_path,
            file_name=task["file_name"],
        )
        if not downloaded:
            continue

        processed = process_pdf(file_path)
        if not processed:
            logger.warning("Processing failed for %s; file retained for retry", file_path.name)
        if PROCESSING_PAUSE_SECONDS > 0:
            logger.info("Pausing %s seconds before next PDF", PROCESSING_PAUSE_SECONDS)
            time.sleep(PROCESSING_PAUSE_SECONDS)
    logger.info("--- Journal task finished ---")
    _release_lock(lock_fd)


def _release_lock(lock_fd):
    """Release the lock file cleanly."""
    try:
        if lock_fd is not None:
            os.close(lock_fd)
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as exc:
        logger.warning("Failed to release lock file: %s", exc)


def main():
    Config.create_directories()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Journal downloader online; process endpoint: %s", PROCESS_ENDPOINT)

    # Immediate run on startup
    run_cycle()

    # Schedule for Mondays at configured time
    schedule.every().monday.at(SCHEDULE_TIME).do(run_cycle)
    logger.info("Next scheduled run: Monday at %s", SCHEDULE_TIME)

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
