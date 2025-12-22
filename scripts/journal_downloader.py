import os
import sys
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import List, Dict, Optional

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
from database.vector_db import VectorDatabase

# Tracking file for processed Mondays (backup in case DB check fails)
PROCESSED_MONDAYS_FILE = Path(os.getenv("PROCESSED_MONDAYS_FILE", "./data/journal_downloads/.processed_mondays.txt"))

# Timezone configuration - Use IST (Indian Standard Time)
IST = ZoneInfo("Asia/Kolkata")

# Configuration (override via env vars)
BASE_URL = os.getenv("JOURNAL_BASE_URL", "https://search.ipindia.gov.in/IPOJournal/Journal/Trademark")
DOWNLOAD_DIR = Path(os.getenv("JOURNAL_DOWNLOAD_DIR", "./data/journal_downloads"))
MAX_WORKERS = int(os.getenv("JOURNAL_MAX_WORKERS", "4"))  # Reserved for future parallelism
SCHEDULE_TIME = os.getenv("JOURNAL_SCHEDULE_TIME", "17:00")  # 24h format
PROCESS_ENDPOINT = os.getenv(
    "PROCESS_PDF_ENDPOINT",
    f"http://{Config.API_HOST}:{Config.API_PORT}/process-pdf-and-index"
)
VERIFY_SSL = os.getenv("JOURNAL_VERIFY_SSL", "false").lower() == "true"
REQUEST_TIMEOUT = int(os.getenv("JOURNAL_REQUEST_TIMEOUT_SECONDS", "60"))
# Processing can take 7–14 minutes; allow a higher timeout for the API call
PROCESS_TIMEOUT = int(os.getenv("JOURNAL_PROCESS_TIMEOUT_SECONDS", "1200"))
PROCESSING_PAUSE_SECONDS = int(os.getenv("JOURNAL_PROCESSING_PAUSE_SECONDS", "30"))
LOCK_FILE = Path(os.getenv("JOURNAL_LOCK_FILE", DOWNLOAD_DIR / ".journal_downloader.lock"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://search.ipindia.gov.in",
    "Referer": "https://search.ipindia.gov.in/IPOJournal/Journal/Trademark",
}

# Ensure log directory exists before logger creation
Config.LOG_PATH.mkdir(parents=True, exist_ok=True)

def get_ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)

logger: logging.Logger = setup_logger(
    "journal_downloader",
    Config.LOG_PATH / f"journal_downloader_{get_ist_now().strftime('%Y%m%d')}.log",
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


def process_pdf(file_path: Path, monday_date: Optional[datetime] = None) -> bool:
    """
    Send the downloaded PDF to the API for processing.
    
    Args:
        file_path: Path to the PDF file
        monday_date: The Monday date this journal belongs to (for report filtering)
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info("Sending for processing: %s", file_path.name)
    
    # Prepare the monday date parameter
    monday_date_str = None
    if monday_date:
        monday_date_str = monday_date.strftime("%Y-%m-%d")
        logger.info("Journal Monday date: %s", monday_date_str)
    
    try:
        with open(file_path, "rb") as pdf_file:
            # Build the request with optional monday date
            files = {"file": (file_path.name, pdf_file, "application/pdf")}
            data = {}
            if monday_date_str:
                data["journal_monday_date"] = monday_date_str
            
            resp = requests.post(
                PROCESS_ENDPOINT,
                files=files,
                data=data,
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


def run_cycle(force_monday: Optional[datetime] = None):
    """
    Download the latest journal PDFs and process them sequentially.
    
    Args:
        force_monday: If provided, process for this specific Monday date.
                     If None, uses the current week's Monday.
    """
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

    # Get the Monday date for this processing cycle
    now = get_ist_now()
    processing_monday = force_monday if force_monday else get_current_monday(now)
    
    logger.info("--- Journal task started ---")
    logger.info("Current time (IST): %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("Processing for Monday: %s", processing_monday.strftime("%Y-%m-%d"))
    
    # Check if this Monday has already been processed
    if is_monday_processed(processing_monday):
        logger.info("Monday %s has already been processed. Skipping.", 
                   processing_monday.strftime("%Y-%m-%d"))
        logger.info("--- Journal task skipped (already processed) ---")
        _release_lock(lock_fd)
        return
    
    tasks = fetch_download_tasks()
    if not tasks:
        logger.info("No tasks found; nothing to download.")
        _release_lock(lock_fd)
        return

    logger.info("Found %d download task(s)", len(tasks))
    
    successful_downloads = 0
    successful_processing = 0
    
    for idx, task in enumerate(tasks, 1):
        file_path: Path = task["file_path"]
        logger.info("Task %d/%d: %s", idx, len(tasks), task["file_name"])
        
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
        
        successful_downloads += 1

        processed = process_pdf(file_path, monday_date=processing_monday)
        if not processed:
            logger.warning("Processing failed for %s; file retained for retry", file_path.name)
        else:
            successful_processing += 1
            # Remove successfully processed PDF to save disk
            try:
                file_path.unlink()
                logger.info("Deleted processed file: %s", file_path.name)
            except Exception as exc:
                logger.warning("Could not delete %s: %s", file_path.name, exc)
        
        if PROCESSING_PAUSE_SECONDS > 0 and idx < len(tasks):
            logger.info("Pausing %s seconds before next PDF", PROCESSING_PAUSE_SECONDS)
            time.sleep(PROCESSING_PAUSE_SECONDS)
    
    # Mark this Monday as processed if we successfully processed at least one PDF
    if successful_processing > 0:
        mark_monday_processed(processing_monday)
        logger.info("Monday %s marked as processed", processing_monday.strftime("%Y-%m-%d"))
    
    logger.info("--- Journal task finished ---")
    logger.info("Summary: %d downloaded, %d processed successfully", successful_downloads, successful_processing)
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


def get_next_monday(from_date: Optional[datetime] = None) -> datetime:
    """
    Calculate the next Monday from the given date in IST timezone.
    If from_date is already Monday, returns the following Monday.
    
    Args:
        from_date: Starting date in IST. If None, uses current IST datetime.
        
    Returns:
        datetime of next Monday at the scheduled time in IST
    """
    if from_date is None:
        from_date = get_ist_now()
    elif from_date.tzinfo is None:
        # If naive datetime, assume it's in IST
        from_date = from_date.replace(tzinfo=IST)
    
    # Parse scheduled time (IST)
    schedule_hour, schedule_minute = map(int, SCHEDULE_TIME.split(':'))
    
    # Get days until next Monday (0 = Monday)
    days_until_monday = (7 - from_date.weekday()) % 7
    
    # If today is Monday but we're past the scheduled time, go to next Monday
    if days_until_monday == 0:
        scheduled_time_today = from_date.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
        if from_date >= scheduled_time_today:
            days_until_monday = 7
    
    next_monday = from_date + timedelta(days=days_until_monday)
    next_monday = next_monday.replace(hour=schedule_hour, minute=schedule_minute, second=0, microsecond=0)
    
    return next_monday


def get_local_time_for_ist_schedule(ist_time_str: str) -> str:
    """
    Convert IST time string to local system time for schedule library.
    The schedule library uses system local time, so we need to convert IST to local.
    
    Args:
        ist_time_str: Time string in HH:MM format (IST timezone)
        
    Returns:
        Time string in HH:MM format (local system timezone)
    """
    # Parse IST time
    hour, minute = map(int, ist_time_str.split(':'))
    
    # Get current date in IST
    ist_now = get_ist_now()
    
    # Create a datetime in IST with the scheduled time (today)
    ist_datetime = ist_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # Convert to local system time
    local_datetime = ist_datetime.astimezone()
    
    # Return as HH:MM string
    return local_datetime.strftime("%H:%M")


def get_current_monday(from_date: Optional[datetime] = None) -> datetime:
    """
    Get the most recent Monday (including today if Monday) in IST timezone.
    
    Args:
        from_date: Reference date in IST. If None, uses current IST datetime.
        
    Returns:
        datetime of the most recent Monday at 00:00:00 IST
    """
    if from_date is None:
        from_date = get_ist_now()
    elif from_date.tzinfo is None:
        # If naive datetime, assume it's in IST
        from_date = from_date.replace(tzinfo=IST)
    
    days_since_monday = from_date.weekday()  # 0 = Monday
    current_monday = from_date - timedelta(days=days_since_monday)
    current_monday = current_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return current_monday


def log_schedule_status():
    """Log the current schedule status with actual dates in IST."""
    now = get_ist_now()
    current_monday = get_current_monday(now)
    next_monday = get_next_monday(now)
    
    logger.info("=" * 50)
    logger.info("Schedule Status (IST):")
    logger.info("  Current time: %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("  This week's Monday: %s", current_monday.strftime("%Y-%m-%d"))
    logger.info("  Next scheduled run (IST): %s", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    
    # Calculate time until next run
    time_until = next_monday - now
    days = time_until.days
    hours, remainder = divmod(time_until.seconds, 3600)
    minutes = remainder // 60
    
    logger.info("  Time until next run: %d days, %d hours, %d minutes", days, hours, minutes)
    logger.info("=" * 50)


def is_monday_processed(monday_date: datetime) -> bool:
    """
    Check if a Monday's journal has already been processed.
    
    First checks the vector database for trademarks with journal_monday_date,
    then falls back to checking the local tracking file.
    
    Args:
        monday_date: The Monday date to check
        
    Returns:
        True if already processed, False otherwise
    """
    monday_str = monday_date.strftime("%Y-%m-%d")
    
    # Method 1: Check vector database
    try:
        logger.info("Checking if Monday %s has been processed (via database)...", monday_str)
        
        # Initialize vector database connection
        if Config.QDRANT_API_KEY:
            qdrant_url = Config.QDRANT_HOST if Config.QDRANT_HOST.startswith('http') else f"https://{Config.QDRANT_HOST}"
            vector_db = VectorDatabase(
                host=qdrant_url,
                port=Config.QDRANT_PORT,
                collection_name=Config.QDRANT_COLLECTION_NAME,
                embedding_dim=Config.EMBEDDING_DIMENSION,
                api_key=Config.QDRANT_API_KEY
            )
        else:
            vector_db = VectorDatabase(
                host=Config.QDRANT_HOST,
                port=Config.QDRANT_PORT,
                collection_name=Config.QDRANT_COLLECTION_NAME,
                embedding_dim=Config.EMBEDDING_DIMENSION,
                api_key=None
            )
        
        # Check if any trademarks exist for this Monday
        trademarks = vector_db.get_trademarks_by_journal_monday(
            monday_date=monday_date,
            include_sources=['pdf_extraction', 'pdf_text_extraction'],
            limit=1  # We only need to know if at least one exists
        )
        
        if trademarks:
            logger.info("Monday %s is ALREADY PROCESSED (%d trademarks found in database)", 
                       monday_str, len(trademarks))
            return True
        else:
            logger.info("Monday %s is NOT YET PROCESSED (no trademarks found in database)", monday_str)
            
    except Exception as e:
        logger.warning("Failed to check database for processed Monday: %s", e)
        logger.info("Falling back to local tracking file...")
    
    # Method 2: Check local tracking file (fallback)
    try:
        if PROCESSED_MONDAYS_FILE.exists():
            processed_dates = PROCESSED_MONDAYS_FILE.read_text().strip().split('\n')
            if monday_str in processed_dates:
                logger.info("Monday %s found in local tracking file (already processed)", monday_str)
                return True
    except Exception as e:
        logger.warning("Failed to read local tracking file: %s", e)
    
    return False


def mark_monday_processed(monday_date: datetime):
    """
    Mark a Monday as processed in the local tracking file.
    
    Args:
        monday_date: The Monday date to mark as processed
    """
    monday_str = monday_date.strftime("%Y-%m-%d")
    
    try:
        PROCESSED_MONDAYS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing dates
        existing_dates = set()
        if PROCESSED_MONDAYS_FILE.exists():
            existing_dates = set(PROCESSED_MONDAYS_FILE.read_text().strip().split('\n'))
            existing_dates.discard('')  # Remove empty strings
        
        # Add new date
        existing_dates.add(monday_str)
        
        # Write back (sorted)
        sorted_dates = sorted(existing_dates)
        PROCESSED_MONDAYS_FILE.write_text('\n'.join(sorted_dates) + '\n')
        
        logger.info("Marked Monday %s as processed in tracking file", monday_str)
        
    except Exception as e:
        logger.warning("Failed to update tracking file: %s", e)


def main():
    Config.create_directories()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Journal Downloader Starting")
    logger.info("=" * 60)
    logger.info("Process endpoint: %s", PROCESS_ENDPOINT)
    logger.info("Download directory: %s", DOWNLOAD_DIR)
    logger.info("Scheduled time (IST): %s (every Monday)", SCHEDULE_TIME)
    
    # Log current schedule status
    log_schedule_status()
    
    now = get_ist_now()
    current_monday = get_current_monday(now)
    
    logger.info("Current week's Monday: %s", current_monday.strftime("%Y-%m-%d"))
    
    # On startup: Check ONLY the current week's Monday
    # Do NOT process older Mondays - only the most recent Monday matters
    if is_monday_processed(current_monday):
        logger.info("Current Monday (%s) is already processed.", current_monday.strftime("%Y-%m-%d"))
        logger.info("Waiting for next Monday's scheduled run.")
    else:
        logger.info("Current Monday (%s) is NOT processed yet.", current_monday.strftime("%Y-%m-%d"))
        logger.info("Processing current Monday's journal now...")
        run_cycle(force_monday=current_monday)

    # Schedule for Mondays at configured time (IST) - convert to local time for schedule library
    local_schedule_time = get_local_time_for_ist_schedule(SCHEDULE_TIME)
    logger.info("IST schedule time: %s, Local schedule time: %s", SCHEDULE_TIME, local_schedule_time)
    schedule.every().monday.at(local_schedule_time).do(run_cycle_with_logging)
    
    # Log next scheduled run
    next_monday = get_next_monday(now)
    logger.info("Next scheduled run (IST): %s", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))

    # Keep running and periodically log status
    last_status_log = get_ist_now()
    while True:
        schedule.run_pending()
        
        # Log schedule status every 6 hours
        if (get_ist_now() - last_status_log).total_seconds() > 6 * 3600:
            log_schedule_status()
            last_status_log = get_ist_now()
        
        time.sleep(30)


def run_cycle_with_logging():
    """
    Wrapper that logs schedule info before running the cycle.
    
    Called automatically by scheduler on Mondays at the scheduled time.
    Processes immediately without delay - the run_cycle function will
    check if already processed and skip if needed.
    """
    now = get_ist_now()
    current_monday = get_current_monday(now)
    
    logger.info("=" * 60)
    logger.info("SCHEDULED RUN TRIGGERED")
    logger.info("Time (IST): %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("Processing Monday: %s", current_monday.strftime("%Y-%m-%d"))
    logger.info("=" * 60)
    
    # Process immediately - run_cycle will check if already processed
    run_cycle()
    
    # Log next scheduled run after completion
    next_monday = get_next_monday(get_ist_now())
    logger.info("Run completed. Next scheduled run: %s (IST)", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))


if __name__ == "__main__":
    main()
