"""
Dashboard Caching Utilities
==========================

Advanced caching system with TTL support for the financial analytics dashboard.

Author: GitHub Copilot
Date: February 8, 2026
"""

import streamlit as st
import time
from functools import wraps
from typing import Dict, Any, Optional, Callable
import threading
from datetime import datetime, timedelta
import json
import hashlib

# Global cache storage
_cache_store: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate a unique cache key from function name and arguments.

    Args:
        func_name: Name of the cached function
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Unique cache key string
    """
    # Convert args and kwargs to a consistent string representation
    args_str = json.dumps(args, sort_keys=True, default=str)
    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

    # Create hash of the combined string
    content = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(content.encode()).hexdigest()

def _is_cache_expired(cache_entry: Dict[str, Any]) -> bool:
    """Check if a cache entry has expired.

    Args:
        cache_entry: Cache entry dictionary

    Returns:
        True if expired, False otherwise
    """
    ttl = cache_entry.get('ttl', 300)  # Default 5 minutes
    created_at = cache_entry.get('created_at', 0)
    return time.time() - created_at > ttl

def _cleanup_expired_cache() -> None:
    """Remove expired entries from the cache."""
    global _cache_store

    with _cache_lock:
        expired_keys = []
        for key, entry in _cache_store.items():
            if _is_cache_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del _cache_store[key]

def cache_with_ttl(ttl_seconds: int = 300, max_size: int = 1000) -> Callable:
    """Decorator to cache function results with time-to-live.

    Args:
        ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)
        max_size: Maximum cache size (default: 1000 entries)

    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _cache_store

            # Generate cache key
            cache_key = _generate_cache_key(func.__name__, args, kwargs)

            # Check cache
            with _cache_lock:
                if cache_key in _cache_store:
                    cache_entry = _cache_store[cache_key]
                    if not _is_cache_expired(cache_entry):
                        # Cache hit
                        cache_entry['hits'] = cache_entry.get('hits', 0) + 1
                        cache_entry['last_accessed'] = time.time()
                        return cache_entry['result']
                    else:
                        # Expired entry
                        del _cache_store[cache_key]

            # Cache miss - execute function
            try:
                result = func(*args, **kwargs)

                # Store in cache
                with _cache_lock:
                    # Check cache size limit
                    if len(_cache_store) >= max_size:
                        # Remove oldest entries (LRU)
                        sorted_entries = sorted(
                            _cache_store.items(),
                            key=lambda x: x[1].get('last_accessed', 0)
                        )
                        # Remove 10% of entries
                        remove_count = max(1, int(max_size * 0.1))
                        for i in range(remove_count):
                            if sorted_entries:
                                del _cache_store[sorted_entries[i][0]]

                    # Store new entry
                    _cache_store[cache_key] = {
                        'result': result,
                        'created_at': time.time(),
                        'last_accessed': time.time(),
                        'ttl': ttl_seconds,
                        'hits': 0,
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }

                return result

            except Exception as e:
                # Don't cache exceptions
                raise e

        return wrapper
    return decorator

def clear_cache_on_demand(pattern: Optional[str] = None) -> Dict[str, int]:
    """Clear cache entries based on a pattern or clear all cache.

    Args:
        pattern: Optional pattern to match function names (default: clear all)

    Returns:
        Dictionary with clearing statistics
    """
    global _cache_store

    with _cache_lock:
        if pattern is None:
            # Clear all cache
            cleared_count = len(_cache_store)
            _cache_store.clear()
            return {
                'cleared_entries': cleared_count,
                'pattern': 'all',
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Clear entries matching pattern
            cleared_count = 0
            keys_to_remove = []

            for key, entry in _cache_store.items():
                func_name = entry.get('function', '')
                if pattern.lower() in func_name.lower():
                    keys_to_remove.append(key)
                    cleared_count += 1

            for key in keys_to_remove:
                del _cache_store[key]

            return {
                'cleared_entries': cleared_count,
                'pattern': pattern,
                'timestamp': datetime.now().isoformat()
            }

def get_cache_info() -> Dict[str, Any]:
    """Get comprehensive information about the current cache state.

    Returns:
        Dictionary with cache statistics and details
    """
    global _cache_store

    with _cache_lock:
        # Clean up expired entries first
        _cleanup_expired_cache()

        total_entries = len(_cache_store)
        total_size = sum(len(str(entry.get('result', ''))) for entry in _cache_store.values())

        # Calculate statistics
        if total_entries > 0:
            hits = [entry.get('hits', 0) for entry in _cache_store.values()]
            ages = [time.time() - entry.get('created_at', 0) for entry in _cache_store.values()]

            stats = {
                'total_entries': total_entries,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'average_hits': round(sum(hits) / len(hits), 2),
                'max_hits': max(hits),
                'min_hits': min(hits),
                'average_age_seconds': round(sum(ages) / len(ages), 2),
                'oldest_entry_seconds': round(max(ages), 2),
                'newest_entry_seconds': round(min(ages), 2),
                'functions_cached': list(set(entry.get('function', 'unknown') for entry in _cache_store.values())),
                'cache_hit_rate': round(sum(hits) / (sum(hits) + total_entries), 3) if total_entries > 0 else 0
            }
        else:
            stats = {
                'total_entries': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'average_hits': 0,
                'max_hits': 0,
                'min_hits': 0,
                'average_age_seconds': 0,
                'oldest_entry_seconds': 0,
                'newest_entry_seconds': 0,
                'functions_cached': [],
                'cache_hit_rate': 0
            }

        # Get top functions by cache usage
        function_usage = {}
        for entry in _cache_store.values():
            func = entry.get('function', 'unknown')
            function_usage[func] = function_usage.get(func, 0) + 1

        stats['top_functions'] = sorted(function_usage.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get cache entries details (limited to avoid memory issues)
        entries_info = []
        for key, entry in list(_cache_store.items())[:50]:  # Limit to first 50 entries
            entries_info.append({
                'function': entry.get('function', 'unknown'),
                'created_at': datetime.fromtimestamp(entry.get('created_at', 0)).isoformat(),
                'hits': entry.get('hits', 0),
                'ttl': entry.get('ttl', 300),
                'expires_in': max(0, entry.get('ttl', 300) - (time.time() - entry.get('created_at', 0)))
            })

        stats['entries_sample'] = entries_info
        stats['timestamp'] = datetime.now().isoformat()

        return stats

# Convenience functions for common cache operations
def clear_data_cache() -> Dict[str, int]:
    """Clear data-related cache entries."""
    return clear_cache_on_demand('data')

def clear_model_cache() -> Dict[str, int]:
    """Clear model-related cache entries."""
    return clear_cache_on_demand('model')

def clear_prediction_cache() -> Dict[str, int]:
    """Clear prediction-related cache entries."""
    return clear_cache_on_demand('predict')

def clear_visualization_cache() -> Dict[str, int]:
    """Clear visualization-related cache entries."""
    return clear_cache_on_demand('chart')

# Cache monitoring functions
def get_cache_hit_rate() -> float:
    """Get current cache hit rate."""
    info = get_cache_info()
    return info.get('cache_hit_rate', 0)

def get_cache_memory_usage() -> float:
    """Get current cache memory usage in MB."""
    info = get_cache_info()
    return info.get('total_size_mb', 0)

def is_cache_healthy() -> bool:
    """Check if cache is operating normally."""
    try:
        info = get_cache_info()
        # Cache is healthy if it has reasonable stats
        return info.get('total_entries', 0) >= 0
    except Exception:
        return False

# Background cache cleanup (optional)
def start_cache_cleanup_scheduler(interval_minutes: int = 30) -> None:
    """Start a background thread to periodically clean up expired cache entries.

    Args:
        interval_minutes: Cleanup interval in minutes
    """
    def cleanup_worker():
        while True:
            time.sleep(interval_minutes * 60)
            try:
                _cleanup_expired_cache()
            except Exception:
                # Silently handle cleanup errors
                pass

    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()