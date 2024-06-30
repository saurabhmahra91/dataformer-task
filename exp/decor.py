"""
Utility decorators for model inference.
"""

import typing
import logging
import functools


def retry(num_retries: int, avoids: tuple[Exception]):
    """
    Try to retry a function if it throws specific set of exceptions.

    Args:
        func (function): The function to retry

        num (int): Number of times of acceptable failures.
            The function will be run num+1 times before raising an exception,
            the first num times all the provided exceptions will be ignored.
            The last (num+1)th time, it will be raised.

        avoids (tuple[Exception]): The exceptions against which the function should be retried.
    """

    def decorator(func: typing.Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(num_retries + 1):
                try:
                    return func(*args, **kwargs)
                except avoids as e:
                    if attempt == num_retries:
                        raise
                    logging.warning("Attempt %d failed for the function %s. Retrying...", attempt+1, func.__name__)
        return wrapper
    return decorator