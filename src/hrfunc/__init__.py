# src/hrfunc/__init__.py

from .hrfunc import montage, estimate_hrfs, localize_hrfs  # or Tool

# Define what's public
__all__ = ["montage", "estimate_hrfs", "localize_hrfs", "tree", "hashtable"]