# src/hrfunc/__init__.py

from .hrfunc import montage, localize_hrfs, load_montage  # or Tool
from .hrtree import tree
from .observer import lens

# Define what's public
__all__ = ["montage", "localize_hrfs", "load_montage", "tree", "lens"]