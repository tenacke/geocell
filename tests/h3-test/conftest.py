"""
Test configuration and fixtures for GEOCELL tests.

This module sets up pytest fixtures and test data used across all test files.
"""

import pytest
import numpy as np

# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def valid_coords():
    """Valid lat/lng coordinates for testing."""
    return [
        (37.3615593, -122.0553238),  # San Francisco area
        (40.7127281, -74.0060152),   # New York
        (51.5074, -0.1278),          # London
        (35.6762, 139.6503),         # Tokyo
        (-33.8688, 151.2093),        # Sydney
        (0.0, 0.0),                  # Origin
        (89.9, 179.9),               # Near north pole, east
        (-89.9, -179.9),             # Near south pole, west
    ]

@pytest.fixture
def valid_coords_array(valid_coords):
    """Valid coordinates as numpy arrays."""
    coords = np.array(valid_coords)
    return coords[:, 0], coords[:, 1]  # lat, lng

@pytest.fixture
def invalid_coords():
    """Invalid lat/lng coordinates for testing."""
    return [
        (91.0, 0.0),      # Latitude too high
        (-91.0, 0.0),     # Latitude too low
        (0.0, 181.0),     # Longitude too high
        (0.0, -181.0),    # Longitude too low
        (100.0, 200.0),   # Both invalid
    ]

@pytest.fixture
def valid_resolutions():
    """Valid H3 resolutions."""
    return [0, 1, 5, 9, 10, 15]

@pytest.fixture
def invalid_resolutions():
    """Invalid H3 resolutions."""
    return [-1, 16, 100]

@pytest.fixture
def valid_cells():
    """Valid H3 cell indices at various resolutions."""
    return {
        0: 0x8029fffffffffff,
        1: 0x81283ffffffffff,
        5: 0x85283473fffffff,
        9: 0x89283082803ffff,
        15: 0x8f283082803abcd,
    }

@pytest.fixture
def pentagon_cells():
    """Known pentagon cell indices."""
    return {
        0: 0x8009fffffffffff,  # Resolution 0 pentagon
        1: 0x81093ffffffffff,  # Resolution 1 pentagon
        5: 0x850d3ffffffffff,  # Resolution 5 pentagon
    }

@pytest.fixture
def invalid_cells():
    """Invalid H3 cell indices."""
    return [
        0,                    # Null
        0xFFFFFFFFFFFFFFFF,  # All bits set
        0x1234567890ABCDEF,  # Random invalid
    ]

@pytest.fixture
def neighbor_pairs():
    """Pairs of neighboring cells."""
    return [
        (0x89283082803ffff, 0x89283082807ffff),
        (0x85283473fffffff, 0x85283447fffffff),
    ]

@pytest.fixture
def non_neighbor_pairs():
    """Pairs of non-neighboring cells."""
    return [
        (0x89283082803ffff, 0x89283080c37ffff),  # Same area, not neighbors
        (0x85283473fffffff, 0x8528342bfffffff),  # Different areas
    ]

# ============================================================================
# COMPARISON HELPERS
# ============================================================================

@pytest.fixture
def assert_close():
    """Helper to assert floating point values are close."""
    def _assert_close(a, b, rtol=1e-9, atol=1e-12):
        if isinstance(a, (list, tuple)):
            a = np.array(a)
        if isinstance(b, (list, tuple)):
            b = np.array(b)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
    return _assert_close

@pytest.fixture
def assert_arrays_equal():
    """Helper to assert arrays are equal."""
    def _assert_equal(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        np.testing.assert_array_equal(a, b)
    return _assert_equal
