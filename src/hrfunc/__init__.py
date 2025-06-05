# src/hrfunc/__init__.py

from .hrfunc import montage, estimate_hrfs, locate_hrfs  # or Tool
from .hrtree import tree

# Define what's public
__all__ = ["montage", "estimate_hrfs", "locate_hrfs", "tree"]