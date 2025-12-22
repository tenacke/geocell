"""
Geocell Cython Extensions.

This package contains Cython implementations for high-performance geospatial operations.

Structure:
----------
- h3/: H3 hexagonal hierarchical geospatial indexing (modular implementation)
  - bindings.pxd: C library declarations
  - common: Shared utilities, constants, exceptions
  - indexing: Coordinate/cell conversions
  - inspection: Cell property queries
  - hierarchy: Parent/child relationships
  - traversal: Grid navigation
  - edges: Directed edge operations
  - vertices: Vertex operations
  - measurements: Distance, area, length calculations

- h3_math_vec.pyx: Vectorized H3 operations
- tile_math.pyx: Tile system operations (planned)
- quadkey_math.pyx: QuadKey operations (planned)
- s2_math.pyx: S2 geometry operations (planned)

Note: These are Cython extension modules (.pyx). They must be compiled to .so files
before they can be imported from Python. Use `make build` to compile.

For Python usage, import from the public API:
    import geocell
    # or
    from geocell.api import h3_api
"""

__all__ = []
