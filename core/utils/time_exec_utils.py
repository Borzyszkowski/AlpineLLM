""" Utilities for the performance logger that measures execution times of functions """

import functools
import logging
import os 
import time


perf_logger = None  # perf_logger is used globally
def initialize_perf_logger(cfg):
    """ Configure the performance logger """
    performance_log_file = os.path.join(cfg.work_dir, 'performance_logs.txt')
    logging.info(f"Initializing the performance logger with output: {performance_log_file}")
    global perf_logger
    perf_logger = logging.getLogger('perf_logger')
    perf_file_handler = logging.FileHandler(performance_log_file)
    perf_file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    perf_file_handler.setFormatter(formatter)
    perf_logger.setLevel(logging.DEBUG)
    perf_logger.addHandler(perf_file_handler)
    return perf_logger


def log_execution_time(func):
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)  # Preserve the original function's metadata
    def wrapper(*args, **kwargs):
        if perf_logger is None:
            return func(*args, **kwargs)
        start_time = time.perf_counter()        # Start timer
        result = func(*args, **kwargs)          # Execute the function
        end_time = time.perf_counter()          # End timer
        execution_time = end_time - start_time
        perf_logger.debug(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper
