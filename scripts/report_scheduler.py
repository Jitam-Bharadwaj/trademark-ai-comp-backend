"""
Report Scheduler - Generates weekly trademark similarity reports

Process:
  1. SYNC self-database trademarks to Qdrant (with CLIP text embeddings)
  2. GENERATE weekly similarity report (comparing journal trademarks against self-DB)

Usage:
  # Run as a scheduled service (every Monday at 7:30 PM)
  python scripts/report_scheduler.py
  
  # Generate report for specific date manually (with self-DB sync)
  python scripts/report_scheduler.py --date 2025-12-15
  
  # Generate report for current Monday manually (no scheduling)
  python scripts/report_scheduler.py --now
  
  # Generate report without syncing self-DB (use existing indexed data)
  python scripts/report_scheduler.py --now --skip-sync

Logic (scheduled mode):
- On startup: Check if current Monday's report exists
  - If NOT exists → sync self-DB, then generate report immediately
  - If exists → wait for next Monday
- On scheduled Monday at 7:30 PM:
  1. Sync self-database to Qdrant (add new marks, remove deleted)
  2. Generate report automatically
- Only the current week's Monday is checked (not older Mondays)
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import schedule

# Ensure project root is on sys.path
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from utils.logger import setup_logger
from database.vector_db import VectorDatabase
from database.application_queries import application_queries
from utils.text_similarity import TextSimilarity
from reporting.report_generator import ReportGenerator
from reporting.pdf_builder import PDFReportBuilder
from models.embedding_model import EmbeddingGenerator

# Import self-database indexing functions
# Handle both cases: running as module or as script
try:
    from scripts.index_self_db import (
        sync_self_db_to_qdrant,
        SELF_DB_SOURCE,
        get_self_db_marks,
        get_existing_self_db_points,
        get_point_id
    )
except ImportError:
    from index_self_db import (
        sync_self_db_to_qdrant,
        SELF_DB_SOURCE,
        get_self_db_marks,
        get_existing_self_db_points,
        get_point_id
    )


# Configuration
SCHEDULE_TIME = os.getenv("REPORT_SCHEDULE_TIME", "19:30")  # 7:30 PM in 24h format
LOCK_FILE = Path(os.getenv("REPORT_LOCK_FILE", "./data/reports/.report_scheduler.lock"))

# Ensure log directory exists
Config.LOG_PATH.mkdir(parents=True, exist_ok=True)

logger: logging.Logger = setup_logger(
    "report_scheduler",
    Config.LOG_PATH / f"report_scheduler_{datetime.now().strftime('%Y%m%d')}.log",
    level=getattr(logging, Config.LOG_LEVEL),
)


def get_current_monday(from_date: Optional[datetime] = None) -> datetime:
    """
    Get the most recent Monday (including today if Monday).
    
    Args:
        from_date: Reference date. If None, uses current datetime.
        
    Returns:
        datetime of the most recent Monday at 00:00:00
    """
    if from_date is None:
        from_date = datetime.now()
    
    days_since_monday = from_date.weekday()  # 0 = Monday
    current_monday = from_date - timedelta(days=days_since_monday)
    current_monday = current_monday.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return current_monday


def get_next_monday(from_date: Optional[datetime] = None) -> datetime:
    """
    Calculate the next Monday from the given date.
    If from_date is Monday and past scheduled time, returns the following Monday.
    
    Args:
        from_date: Starting date. If None, uses current datetime.
        
    Returns:
        datetime of next Monday at the scheduled time
    """
    if from_date is None:
        from_date = datetime.now()
    
    # Parse scheduled time
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


def is_report_generated(monday_date: datetime) -> bool:
    """
    Check if a report has already been generated for a specific Monday.
    
    Checks the reports directory for files matching the Monday date pattern.
    
    Args:
        monday_date: The Monday date to check
        
    Returns:
        True if report exists, False otherwise
    """
    monday_str = monday_date.strftime("%Y%m%d")
    reports_dir = Config.REPORT_OUTPUT_PATH
    
    if not reports_dir.exists():
        logger.info("Reports directory does not exist yet: %s", reports_dir)
        return False
    
    # Look for report files matching this Monday's date pattern
    # Report files are named: report_YYYYMMDD_xxxxxxxx.pdf
    pattern = f"report_{monday_str}_*.pdf"
    matching_files = list(reports_dir.glob(pattern))
    
    if matching_files:
        logger.info("Report for Monday %s already exists: %s", 
                   monday_date.strftime("%Y-%m-%d"), matching_files[0].name)
        return True
    
    logger.info("No report found for Monday %s", monday_date.strftime("%Y-%m-%d"))
    return False


def init_components():
    """
    Initialize all required components for report generation.
    
    Returns:
        Tuple of (vector_db, text_similarity, report_generator, pdf_builder)
    """
    # Initialize vector database
    logger.info("Connecting to vector database...")
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
    logger.info("Connected to vector database")
    
    # Initialize text similarity
    logger.info("Initializing text similarity...")
    text_similarity = TextSimilarity(
        use_levenshtein=True,
        use_phonetic=True,
        use_fuzzywuzzy=True
    )
    
    # Initialize report generator
    logger.info("Initializing report generator...")
    report_generator = ReportGenerator(vector_db=vector_db, text_similarity=text_similarity)
    
    # Initialize PDF builder
    logger.info("Initializing PDF builder...")
    pdf_builder = PDFReportBuilder()
    
    return vector_db, text_similarity, report_generator, pdf_builder


def generate_report(monday_date: datetime, verbose: bool = False) -> Optional[Path]:
    """
    Generate a weekly similarity report for the specified Monday.
    
    Args:
        monday_date: The Monday date for the report
        verbose: If True, print detailed output to console
        
    Returns:
        Path to the generated PDF report, or None if failed
    """
    def log_and_print(msg, *args):
        """Helper to both log and optionally print"""
        formatted = msg % args if args else msg
        logger.info(msg, *args) if args else logger.info(msg)
        if verbose:
            print(formatted)
    
    log_and_print("=" * 60)
    log_and_print("Weekly Trademark Similarity Report Generator")
    log_and_print("=" * 60)
    log_and_print("Similarity Threshold: 0.5 (50%%)")
    log_and_print("Report Date: %s", monday_date.strftime("%A, %B %d, %Y"))
    log_and_print("=" * 60)
    
    try:
        # Initialize components
        log_and_print("\nInitializing components...")
        _, _, report_generator, pdf_builder = init_components()
        log_and_print("Components initialized successfully")
        
        # Generate the report
        log_and_print("\nGenerating report data...")
        log_and_print("-" * 40)
        report = report_generator.generate_weekly_report(monday_date=monday_date)
        log_and_print("-" * 40)
        
        # Log summary
        log_and_print("\nReport Summary:")
        log_and_print("  - Report ID: %s", report.report_id)
        log_and_print("  - Journal Trademarks: %d", report.summary.total_journal_trademarks)
        log_and_print("    - Image-based: %d", report.summary.total_image_based)
        log_and_print("    - Text-only: %d", report.summary.total_text_only)
        log_and_print("  - Self DB Trademarks: %d", report.summary.total_self_db_trademarks)
        log_and_print("  - Total Similarities: %d", report.summary.total_similarities_found)
        log_and_print("  - Trademarks with Matches: %d", report.summary.trademarks_with_similarities)
        if report.summary.highest_similarity_score > 0:
            log_and_print("  - Highest Similarity: %.2f%%", report.summary.highest_similarity_score * 100)
            log_and_print("  - Average Similarity: %.2f%%", report.summary.average_similarity_score * 100)
        log_and_print("  - Processing Time: %.2fs", report.processing_time_seconds)
        
        # Build PDF
        log_and_print("\nBuilding PDF report...")
        pdf_path = pdf_builder.build_report(report)
        
        log_and_print("\n" + "=" * 60)
        log_and_print("Report generated successfully!")
        log_and_print("PDF saved to: %s", pdf_path)
        log_and_print("=" * 60)
        
        return pdf_path
        
    except Exception as e:
        logger.error("Failed to generate report: %s", e, exc_info=True)
        if verbose:
            print(f"\nError: Failed to generate report: {e}")
        return None


def sync_self_database(vector_db: VectorDatabase, verbose: bool = False) -> bool:
    """
    Sync self-database trademarks to Qdrant before report generation.
    
    This ensures the self-database marks are up-to-date with CLIP text embeddings
    for cross-modal similarity comparison with journal image trademarks.
    
    Args:
        vector_db: VectorDatabase instance
        verbose: If True, print detailed output to console
        
    Returns:
        True if sync succeeded, False otherwise
    """
    def log_and_print(msg, *args):
        """Helper to both log and optionally print"""
        formatted = msg % args if args else msg
        logger.info(msg, *args) if args else logger.info(msg)
        if verbose:
            print(formatted)
    
    log_and_print("=" * 60)
    log_and_print("STEP 1: Syncing Self-Database to Qdrant")
    log_and_print("=" * 60)
    
    try:
        # Initialize embedding generator for CLIP text embeddings
        log_and_print("Initializing CLIP embedding generator...")
        embedding_generator = EmbeddingGenerator()
        log_and_print("CLIP model loaded (embedding dim: %d)", embedding_generator.embedding_dim)
        
        # Sync self-database marks to Qdrant
        log_and_print("\nSyncing self-database marks...")
        stats = sync_self_db_to_qdrant(
            vector_db=vector_db,
            embedding_generator=embedding_generator,
            force=False,  # Incremental sync (add new, delete removed)
            dry_run=False
        )
        
        log_and_print("\nSelf-Database Sync Complete:")
        log_and_print("  - Added:     %d", stats.get('added', 0))
        log_and_print("  - Deleted:   %d", stats.get('deleted', 0))
        log_and_print("  - Unchanged: %d", stats.get('unchanged', 0))
        log_and_print("  - Failed:    %d", stats.get('failed', 0))
        log_and_print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error("Failed to sync self-database: %s", e, exc_info=True)
        if verbose:
            print(f"\nError: Failed to sync self-database: {e}")
        return False


def run_cycle(force_monday: Optional[datetime] = None):
    """
    Sync self-database and generate the weekly report if not already done.
    
    Process:
    1. First, sync self-database trademarks to Qdrant (for CLIP cross-modal similarity)
    2. Then, generate the weekly similarity report
    
    Args:
        force_monday: If provided, generate for this specific Monday.
                     If None, uses the current week's Monday.
    """
    # Prevent overlapping runs
    lock_fd = None
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        logger.info("Another report generation is already in progress; skipping.")
        return
    except Exception as exc:
        logger.error("Failed to acquire lock: %s", exc)
        return
    
    now = datetime.now()
    processing_monday = force_monday if force_monday else get_current_monday(now)
    
    logger.info("=" * 60)
    logger.info("--- Weekly Report Generation Process Started ---")
    logger.info("=" * 60)
    logger.info("Current time: %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("Processing for Monday: %s", processing_monday.strftime("%Y-%m-%d"))
    
    # Check if report already exists
    if is_report_generated(processing_monday):
        logger.info("Report for Monday %s already exists. Skipping.", 
                   processing_monday.strftime("%Y-%m-%d"))
        logger.info("--- Report generation task skipped (already exists) ---")
        _release_lock(lock_fd)
        return
    
    # STEP 1: Sync self-database to Qdrant
    logger.info("")
    logger.info("STEP 1: Syncing self-database trademarks to Qdrant...")
    
    # Initialize vector database for sync
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
    
    sync_success = sync_self_database(vector_db, verbose=False)
    
    if not sync_success:
        logger.warning("Self-database sync had issues, but continuing with report generation...")
    
    # STEP 2: Generate the report
    logger.info("")
    logger.info("STEP 2: Generating weekly similarity report...")
    pdf_path = generate_report(processing_monday, verbose=False)
    
    if pdf_path:
        logger.info("=" * 60)
        logger.info("--- Weekly Report Generation Process Completed Successfully ---")
        logger.info("=" * 60)
    else:
        logger.error("--- Report generation task failed ---")
    
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


def run_cycle_with_logging():
    """
    Wrapper that logs schedule info before running the cycle.
    
    Called automatically by scheduler on Mondays at the scheduled time.
    """
    now = datetime.now()
    current_monday = get_current_monday(now)
    
    logger.info("=" * 60)
    logger.info("SCHEDULED REPORT GENERATION TRIGGERED")
    logger.info("Time: %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("Report for Monday: %s", current_monday.strftime("%Y-%m-%d"))
    logger.info("=" * 60)
    
    # Generate immediately - run_cycle will check if already exists
    run_cycle()
    
    # Log next scheduled run
    next_monday = get_next_monday(datetime.now())
    logger.info("Completed. Next scheduled run: %s", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))


def log_schedule_status():
    """Log the current schedule status with actual dates."""
    now = datetime.now()
    current_monday = get_current_monday(now)
    next_monday = get_next_monday(now)
    
    logger.info("=" * 50)
    logger.info("Report Schedule Status:")
    logger.info("  Current time: %s", now.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    logger.info("  This week's Monday: %s", current_monday.strftime("%Y-%m-%d"))
    logger.info("  Next scheduled run: %s", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    
    # Calculate time until next run
    time_until = next_monday - now
    days = time_until.days
    hours, remainder = divmod(time_until.seconds, 3600)
    minutes = remainder // 60
    
    logger.info("  Time until next run: %d days, %d hours, %d minutes", days, hours, minutes)
    logger.info("=" * 50)


def run_scheduled_service():
    """Run as a scheduled service (daemon mode)."""
    logger.info("=" * 60)
    logger.info("Report Scheduler Service Starting")
    logger.info("=" * 60)
    logger.info("Reports directory: %s", Config.REPORT_OUTPUT_PATH)
    logger.info("Scheduled time: %s (every Monday)", SCHEDULE_TIME)
    
    # Log current schedule status
    log_schedule_status()
    
    now = datetime.now()
    current_monday = get_current_monday(now)
    
    logger.info("Current week's Monday: %s", current_monday.strftime("%Y-%m-%d"))
    
    # On startup: Check ONLY the current week's Monday
    if is_report_generated(current_monday):
        logger.info("Current Monday (%s) report already exists.", current_monday.strftime("%Y-%m-%d"))
        logger.info("Waiting for next Monday's scheduled run.")
    else:
        logger.info("Current Monday (%s) report does NOT exist.", current_monday.strftime("%Y-%m-%d"))
        logger.info("Generating report now...")
        run_cycle(force_monday=current_monday)
    
    # Schedule for Mondays at 7:30 PM
    schedule.every().monday.at(SCHEDULE_TIME).do(run_cycle_with_logging)
    
    # Log next scheduled run
    next_monday = get_next_monday(now)
    logger.info("Next scheduled run: %s", next_monday.strftime("%Y-%m-%d %H:%M:%S (%A)"))
    
    # Keep running and periodically log status
    last_status_log = datetime.now()
    while True:
        schedule.run_pending()
        
        # Log schedule status every 6 hours
        if (datetime.now() - last_status_log).total_seconds() > 6 * 3600:
            log_schedule_status()
            last_status_log = datetime.now()
        
        time.sleep(30)


def run_manual_generation(date_str: Optional[str] = None, skip_sync: bool = False):
    """
    Run manual report generation (one-time, no scheduling).
    
    Process:
    1. First, sync self-database trademarks to Qdrant (unless skip_sync=True)
    2. Then, generate the weekly similarity report

    Args:
        date_str: Date in YYYY-MM-DD format, or None for current Monday
        skip_sync: If True, skip self-database sync (use existing indexed data)
    """
    Config.create_directories()

    # Parse date if provided
    if date_str:
        try:
            monday_date = datetime.strptime(date_str, "%Y-%m-%d")
            if monday_date.weekday() != 0:
                print(f"Warning: {date_str} is not a Monday, but proceeding anyway...")
        except ValueError:
            print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        # Calculate most recent Monday
        monday_date = get_current_monday()

    monday_date = monday_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # STEP 1: Sync self-database to Qdrant (unless skipped)
    if not skip_sync:
        print("\n" + "=" * 60)
        print("STEP 1: Syncing Self-Database to Qdrant")
        print("=" * 60)
        
        # Initialize vector database for sync
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
        
        sync_success = sync_self_database(vector_db, verbose=True)
        
        if not sync_success:
            print("Warning: Self-database sync had issues, but continuing with report generation...")
    else:
        print("\nSkipping self-database sync (using existing indexed data)")

    # STEP 2: Generate report with verbose output
    print("\n" + "=" * 60)
    print("STEP 2: Generating Weekly Similarity Report")
    print("=" * 60)
    
    pdf_path = generate_report(monday_date, verbose=True)

    if pdf_path:
        return pdf_path
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Weekly Trademark Similarity Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run as scheduled service (every Monday at 7:30 PM)
  python scripts/report_scheduler.py
  
  # Generate report for specific date (with self-DB sync)
  python scripts/report_scheduler.py --date 2025-12-15
  
  # Generate report for current Monday (one-time, with sync)
  python scripts/report_scheduler.py --now
  
  # Generate report without syncing self-DB (use existing indexed data)
  python scripts/report_scheduler.py --now --skip-sync
        """
    )
    
    parser.add_argument(
        '--date', 
        type=str, 
        help='Generate report for specific date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--now', 
        action='store_true',
        help='Generate report for current Monday immediately (no scheduling)'
    )
    parser.add_argument(
        '--skip-sync', 
        action='store_true',
        help='Skip self-database sync before report generation'
    )
    
    args = parser.parse_args()
    
    # Create directories
    Config.create_directories()
    
    if args.date:
        # Manual generation for specific date
        print(f"Manual report generation for date: {args.date}")
        run_manual_generation(args.date, skip_sync=args.skip_sync)
    elif args.now:
        # Manual generation for current Monday
        print("Manual report generation for current Monday")
        run_manual_generation(None, skip_sync=args.skip_sync)
    else:
        # Run as scheduled service
        print("Starting Report Scheduler Service...")
        print(f"Reports will be generated every Monday at {SCHEDULE_TIME}")
        run_scheduled_service()


if __name__ == "__main__":
    main()

