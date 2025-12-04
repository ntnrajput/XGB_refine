# utils/tracer.py - Execution flow tracer

import functools
import inspect
from datetime import datetime
from pathlib import Path


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


# Track call stack depth
_call_depth = 0


    """
    Trace function calls.
    
    Args:
        file_name: Name of the file
        func_name: Name of the function
        event: ENTER or EXIT
    """
    global _call_depth
    
    indent = "  " * _call_depth
    
    if event == "ENTER":
        print(f"{Colors.CYAN}{indent}→ [{file_name}::{func_name}] Entering{Colors.END}")
        _call_depth += 1
    elif event == "EXIT":
        _call_depth = max(0, _call_depth - 1)
        indent = "  " * _call_depth
        print(f"{Colors.GREEN}{indent}← [{file_name}::{func_name}] Exiting{Colors.END}")


def trace_print(file_name: str, *args, **kwargs):
    """
    Traced print function.
    
    Args:
        file_name: Name of the file
        *args: Print arguments
        **kwargs: Print keyword arguments
    """
    global _call_depth
    
    indent = "  " * _call_depth
    
    # Get the line number from call stack
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    line_no = caller_frame.f_lineno if caller_frame else "?"
    
    # Format message
    prefix = f"{Colors.YELLOW}{indent}[{file_name}:{line_no}]{Colors.END} "
    
    # Print with prefix
    print(prefix, *args, **kwargs)


def trace_decorator(func):
    """
    Decorator to automatically trace function execution.
    
    Usage:
        @trace_decorator
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        file_name = Path(inspect.getfile(func)).name
        func_name = func.__name__
        
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            raise
    
    return wrapper


__all__ = ['trace_call', 'trace_print', 'trace_decorator', 'Colors']
