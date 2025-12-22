"""GEOCELL: High-performance H3 geospatial indexing library."""

from geocell._version import __version__

__all__ = ["__version__"]

def version() -> str:
    """Get the current version of the geocell package."""
    return __version__
