# log_config.py
import logging
import sys
import os
from pathlib import Path


output_dir = os.environ.get('JOB_OUTPUT_DIR', os.getcwd());
Path(output_dir).mkdir(parents=True, exist_ok=True)



print(f"Output dir: {output_dir}")

# --- Configuration Variables ---
ERROR_LOG_FILENAME = os.path.join(output_dir, 'script_error_log.log') # Define a log file name
STATUS_LOG_FILENAME = os.path.join(output_dir, 'script_status_log.log') # Define a log file name
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
APP_LOG_LEVEL = logging.DEBUG # Or logging.INFO
# Define a main logger name for your application
APP_LOGGER_NAME = 'my_app'

# --- Setup Logging ---

# Get the logger by the specific application name
logger = logging.getLogger(APP_LOGGER_NAME)
logger.setLevel(APP_LOG_LEVEL)
logger.propagate = False # Important to prevent duplicate logs if root is configured

# Check if handlers already exist (prevents adding handlers multiple times
# if this module is somehow imported again, though Python usually caches imports)
if not logger.handlers:
    # Create Handlers
    status_formatter = logging.Formatter(LOG_FORMAT)
    status_file_handler = logging.FileHandler(STATUS_LOG_FILENAME, mode='a+')
    status_file_handler.setLevel(logging.INFO)
    status_file_handler.setFormatter(status_formatter)

    error_formatter = logging.Formatter(LOG_FORMAT)
    error_file_handler = logging.FileHandler(ERROR_LOG_FILENAME, mode='a+')
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(error_formatter)

    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) # Adjust as needed
    console_handler.setFormatter(console_formatter)

    # Add Handlers
    logger.addHandler(status_file_handler)
    logger.addHandler(error_file_handler)
    # logger.addHandler(console_handler)

    logger.info("Logging configured from log_config.py") # Optional: Log that setup happened

# --- Example Usage ---

# logger.info("Application starting up. Goes to status.log.")
# logger.warning("Something might be wrong. Goes to status.log .")
# logger.error("An error occurred! Goes to status.log, and error.log.")
# logger.critical("A critical failure! Goes to status.log, and error.log.")

# print(f"\nCheck '{STATUS_LOG_FILENAME}' and '{ERROR_LOG_FILENAME}' for output.")
