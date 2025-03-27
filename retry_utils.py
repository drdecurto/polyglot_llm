# retry_utils.py - Updated with longer delays and more retries
# Dr. de Curtò ; BARCELONA Supercomputing Center / Universidad Pontificia Comillas / UOC
# Dr. de Zarzà; Universidad de Zaragoza / UOC

import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Type, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retry_utils")

class RetryException(Exception):
    """Exception indicating that all retry attempts failed"""
    pass

def retry_with_backoff(
    max_retries: int = 5,               # Increased from 3 to 5
    initial_delay: float = 5.0,         # Increased from 2.0 to 5.0
    max_delay: float = 120.0,           # Increased from 60.0 to 120.0
    backoff_factor: float = 2.5,        # Increased from 2.0 to 2.5
    jitter: bool = True,
    exceptions_to_retry: Optional[List[Type[Exception]]] = None
) -> Callable:
    """
    Decorator that implements exponential backoff retries
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases exponentially
        jitter: Whether to add randomness to the delay to prevent thundering herd
        exceptions_to_retry: List of exception types to retry on (defaults to all exceptions)
    
    Returns:
        Decorated function with retry logic
    """
    if exceptions_to_retry is None:
        exceptions_to_retry = [Exception]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for the initial try
                try:
                    if attempt > 0:
                        # Only log for retry attempts, not the initial attempt
                        logger.info(
                            f"Retry attempt {attempt}/{max_retries} for {func.__name__} after {delay:.2f}s delay"
                        )
                        
                    return func(*args, **kwargs)
                    
                except tuple(exceptions_to_retry) as e:
                    last_exception = e
                    
                    # Check if we've reached max retries
                    if attempt == max_retries:
                        logger.warning(
                            f"Maximum retries ({max_retries}) reached for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
                        break
                    
                    # Calculate next backoff delay
                    if jitter:
                        # Add random jitter between 0% to 25% of the current delay
                        jitter_amount = random.uniform(0, 0.25 * delay)
                        actual_delay = delay + jitter_amount
                    else:
                        actual_delay = delay
                    
                    logger.info(
                        f"Encountered error: {str(e)}. "
                        f"Waiting {actual_delay:.2f}s before next retry."
                    )
                    
                    # Sleep before next attempt
                    time.sleep(actual_delay)
                    
                    # Increase delay for next attempt using exponential backoff
                    delay = min(delay * backoff_factor, max_delay)
            
            # If we get here, all retries failed
            raise RetryException(
                f"Function {func.__name__} failed after {max_retries} retries. "
                f"Last error: {str(last_exception)}"
            ) from last_exception
        
        return wrapper
    
    return decorator

def retry_api_call(
    api_func: Callable,
    *args: Any,
    max_retries: int = 5,               # Increased from 3 to 5
    initial_delay: float = 5.0,         # Increased from 1.0 to 5.0
    max_delay: float = 120.0,           # Increased from 60.0 to 120.0 
    backoff_factor: float = 2.5,        # Increased from 2.0 to 2.5
    jitter: bool = True,
    exceptions_to_retry: Optional[List[Type[Exception]]] = None,
    **kwargs: Any
) -> Any:
    """
    Utility function to retry an API call with exponential backoff.
    
    Args:
        api_func: Function to call
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases exponentially
        jitter: Whether to add randomness to the delay
        exceptions_to_retry: List of exception types to retry on
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Result of the API call if successful
        
    Raises:
        RetryException: If all retry attempts fail
    """
    if exceptions_to_retry is None:
        exceptions_to_retry = [Exception]
        
    @retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        exceptions_to_retry=exceptions_to_retry
    )
    def _wrapped_api_call() -> Any:
        return api_func(*args, **kwargs)
    
    return _wrapped_api_call()