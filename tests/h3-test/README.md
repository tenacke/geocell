# GEOCELL Test Suite

This directory contains comprehensive tests for the GEOCELL library, comparing against the official H3 library to ensure correctness.

## Test Structure

### Test Files

- **`conftest.py`**: Pytest fixtures and shared test data
  - Valid/invalid coordinates (8 locations worldwide + edge cases)
  - Valid/invalid resolutions
  - Cell test data (regular cells, pentagons, invalid cells)
  - Neighbor relationships
  - Comparison helper functions

- **`test_indexing.py`**: Coordinate/cell conversion tests
  - `lat_lng_to_cell`, `cell_to_lat_lng`, `cell_to_boundary`
  - Scalar and vectorized operations
  - Invalid input handling
  - Roundtrip testing

- **`test_inspection.py`**: Cell property tests
  - `get_resolution`, `is_valid_cell`, `is_pentagon`
  - `is_res_class_III`, `get_base_cell_number`
  - `get_icosahedron_faces`
  - Scalar and vectorized operations

- **`test_hierarchical.py`**: Parent/child relationship tests
  - `cell_to_parent`, `cell_to_children`, `cell_to_children_size`
  - `cell_to_center_child`
  - `compact_cells`, `uncompact_cells`
  - Roundtrip testing

- **`test_traversal.py`**: Grid traversal tests
  - `grid_disk`, `grid_ring`, `grid_distance`, `grid_path_cells`
  - `are_neighbor_cells`, `cells_to_directed_edge`
  - Scalar and vectorized operations
  - Pentagon handling

- **`test_edges.py`**: Edge and vertex tests
  - Directed edge operations
  - `origin_to_directed_edges`, `directed_edge_to_boundary`
  - `cell_to_vertex`, `cell_to_vertexes`, `vertex_to_lat_lng`
  - `is_valid_directed_edge`, `is_valid_vertex`

- **`test_measurements.py`**: Area, length, distance tests
  - `get_hexagon_area_avg_*`, `cell_area_*`
  - `get_hexagon_edge_length_avg_*`, `edge_length_*`
  - `great_circle_distance_*`
  - `get_num_cells`, `get_res0_cells`, `get_pentagons`
  - Different unit systems (km, m, rads)

- **`test_errors.py`**: Error handling tests
  - Invalid coordinates (out of range, NaN, inf)
  - Invalid resolutions (negative, too high)
  - Invalid cells (zero, max uint64, random)
  - Hierarchical errors (wrong resolution relationships)
  - Traversal errors (negative k, different resolutions)
  - Edge errors (invalid edges, out of range vertices)
  - Vectorized errors (mismatched sizes, empty arrays)
  - Type errors (wrong types, None values)

## Testing Strategy

### Comparison with Official H3

All tests compare GEOCELL results against the official H3 library:

```python
# Official H3
h3_result = h3.some_function(args)

# GEOCELL
geocell_result = geocell.some_function(args)

# Assert they match
assert geocell_result == h3_result
```

### Success Cases
- Results must match H3 exactly (for integers/cells)
- Floating point values use `np.testing.assert_allclose` with tight tolerances (rtol=1e-9)

### Error Cases
- Both libraries must raise exceptions for the same invalid inputs
- Error types don't need to match exactly, but both must fail

### Vectorized Operations

Since H3 doesn't support vectorized operations natively, expected values are computed with loops:

```python
# H3 doesn't have vectorization, compute with loop
expected = np.array([
    h3.some_function(values[i])
    for i in range(len(values))
])

# GEOCELL vectorized
result = geocell.some_function(values)

# Compare
np.testing.assert_array_equal(result, expected)
```

## Running Tests

### Run all tests
```bash
make test
# or
pytest tests/ -v
```

### Run specific test file
```bash
make test-indexing
# or
pytest tests/test_indexing.py -v
```

### Run specific test class
```bash
pytest tests/test_indexing.py::TestLatLngToCell -v
```

### Run specific test
```bash
pytest tests/test_indexing.py::TestLatLngToCell::test_scalar_success -v
```

### Run fast tests only (exclude slow)
```bash
make test-fast
# or
pytest tests/ -v -m "not slow"
```

### Run with coverage
```bash
make coverage
# or
pytest tests/ --cov=geocell --cov-report=html
```

## Test Data Coverage

### Coordinates
- **San Francisco**: 37.36, -122.06 (tech hub)
- **New York**: 40.71, -74.01 (dense urban)
- **London**: 51.50, -0.12 (Europe)
- **Tokyo**: 35.67, 139.65 (Asia)
- **Sydney**: -33.86, 151.20 (Southern hemisphere)
- **Origin**: 0, 0 (equator/prime meridian)
- **North Pole**: 89.9, 179.9 (near pole)
- **South Pole**: -89.9, -179.9 (near pole)

### Resolutions
- **0**: Coarsest (122 base cells)
- **1**: Continental
- **5**: City
- **9**: Neighborhood
- **10**: Block
- **15**: Finest (~0.5m)

### Special Cases
- **Pentagons**: 12 per resolution (base cell centers)
- **Hexagons**: All other cells
- **Edge cases**: Poles, dateline, equator

## Pre-Build Requirement

Tests must pass before building the library:

```bash
# This runs tests first, then builds
make build

# To skip tests (use with caution)
make build-force
```

## Test Fixtures

All test fixtures are defined in `conftest.py`:

- `valid_coords`: List of (lat, lng) tuples
- `valid_coords_array`: NumPy arrays of lat/lng
- `invalid_coords`: Out of range coordinates
- `valid_resolutions`: [0, 1, 5, 9, 10, 15]
- `invalid_resolutions`: [-1, 16, 100]
- `valid_cells`: Dict of {resolution: cell}
- `pentagon_cells`: Dict of {resolution: pentagon_cell}
- `invalid_cells`: List of invalid cell values
- `neighbor_pairs`: Pairs of neighboring cells
- `non_neighbor_pairs`: Pairs of non-neighboring cells
- `assert_close`: Helper for floating point comparison
- `assert_arrays_equal`: Helper for array comparison

## Dependencies

Tests require:
- `pytest>=8.0.0`: Test framework
- `h3>=4.0.0`: Official H3 library for comparison
- `numpy>=2.3.5`: Array operations
- All GEOCELL dependencies

Install with:
```bash
pip install -e ".[dev]"
# or
make dev-install
```
