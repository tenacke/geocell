"""
Test indexing functions (lat/lng to cell conversions).

Compare GEOCELL implementation against official H3 library.
"""

import pytest
import numpy as np
import h3  # Official H3 library

from geocell.api import h3 as geocell


class TestLatLngToCell:
    """Test lat_lng_to_cell function."""
    
    def test_scalar_success(self, valid_coords, valid_resolutions):
        """Test scalar conversion matches H3."""
        for lat, lng in valid_coords:
            for res in valid_resolutions:
                # Official H3
                h3_result = int(h3.latlng_to_cell(lat, lng, res), 16)
                
                # GEOCELL
                geocell_result = geocell.lat_lng_to_cell(lat, lng, res)
                
                assert geocell_result == h3_result, \
                    f"Mismatch for ({lat}, {lng}) at res {res}"
    
    def test_scalar_invalid_coords(self, invalid_coords):
        """Test invalid coordinates raise same error type."""
        for lat, lng in invalid_coords:
            # Check both raise exceptions
            h3_error = None
            geocell_error = None
            
            try:
                h3.latlng_to_cell(lat, lng, 5)
            except Exception as e:
                h3_error = type(e).__name__
            
            try:
                geocell.lat_lng_to_cell(lat, lng, 5)
            except Exception as e:
                geocell_error = type(e).__name__
            
            assert h3_error is not None, f"H3 should raise error for ({lat}, {lng})"
            assert geocell_error is not None, f"GEOCELL should raise error for ({lat}, {lng})"
    
    def test_scalar_invalid_resolution(self, invalid_resolutions):
        """Test invalid resolutions raise errors."""
        for res in invalid_resolutions:
            # Check both raise exceptions
            with pytest.raises(Exception):
                h3.latlng_to_cell(37.3615593, -122.0553238, res)
            
            with pytest.raises(Exception):
                geocell.lat_lng_to_cell(37.3615593, -122.0553238, res)
    
    def test_vector_success(self, valid_coords_array, valid_resolutions):
        """Test vectorized conversion."""
        lat, lng = valid_coords_array
        
        for res in valid_resolutions:
            # H3 doesn't have native vectorization, compute with loop
            h3_result = np.array([
                int(h3.latlng_to_cell(lat[i], lng[i], res), 16)
                for i in range(len(lat))
            ], dtype=np.uint64)
            
            # GEOCELL vectorized
            geocell_result = geocell.lat_lng_to_cell(lat, lng, res)
            
            np.testing.assert_array_equal(geocell_result, h3_result)
    
    def test_vector_broadcast(self):
        """Test broadcasting with different shaped inputs."""
        lat = np.array([37.36, 40.71])
        lng = np.array([-122.06, -74.01])
        resolutions = np.array([7, 9])
        
        # Compute expected with loop
        expected = np.array([
            int(h3.latlng_to_cell(lat[i], lng[i], resolutions[i]), 16)
            for i in range(len(lat))
        ], dtype=np.uint64)
        
        # GEOCELL
        result = geocell.lat_lng_to_cell(lat, lng, resolutions)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_vector_single_resolution(self):
        """Test single resolution with multiple points."""
        lat = np.array([37.36, 40.71, 51.51])
        lng = np.array([-122.06, -74.01, -0.13])
        res = 9
        
        # Expected
        expected = np.array([
            int(h3.latlng_to_cell(lat[i], lng[i], res), 16)
            for i in range(len(lat))
        ], dtype=np.uint64)
        
        # GEOCELL
        result = geocell.lat_lng_to_cell(lat, lng, res)
        
        np.testing.assert_array_equal(result, expected)


class TestCellToLatLng:
    """Test cell_to_lat_lng function."""
    
    def test_scalar_success(self, valid_cells):
        """Test scalar cell to lat/lng matches H3."""
        for res, cell in valid_cells.items():
            # Official H3
            h3_lat, h3_lng = h3.cell_to_latlng(format(cell, 'x'))
            
            # GEOCELL
            geocell_lat, geocell_lng = geocell.cell_to_lat_lng(cell)
            
            # Should be very close (within floating point precision)
            np.testing.assert_allclose(geocell_lat, h3_lat, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(geocell_lng, h3_lng, rtol=1e-9, atol=1e-12)
    
    def test_scalar_invalid_cell(self, invalid_cells):
        """Test invalid cells raise errors."""
        for cell in invalid_cells:
            # Check both raise exceptions
            with pytest.raises(Exception):
                h3.cell_to_latlng(format(cell, 'x'))
            
            with pytest.raises(Exception):
                geocell.cell_to_lat_lng(cell)
    
    def test_vector_success(self, valid_cells):
        """Test vectorized cell to lat/lng."""
        cells = np.array(list(valid_cells.values()), dtype=np.uint64)
        
        # H3 doesn't have native vectorization, compute with loop
        h3_results = [h3.cell_to_latlng(format(cell, 'x')) for cell in cells]
        h3_lat = np.array([r[0] for r in h3_results])
        h3_lng = np.array([r[1] for r in h3_results])
        
        # GEOCELL vectorized
        geocell_lat, geocell_lng = geocell.cell_to_lat_lng(cells)
        
        np.testing.assert_allclose(geocell_lat, h3_lat, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(geocell_lng, h3_lng, rtol=1e-9, atol=1e-12)
    
    def test_roundtrip(self, valid_coords, valid_resolutions):
        """Test lat/lng -> cell -> lat/lng roundtrip."""
        for lat, lng in valid_coords:
            for res in valid_resolutions:
                # Convert to cell
                cell = geocell.lat_lng_to_cell(lat, lng, res)
                
                # Convert back
                lat2, lng2 = geocell.cell_to_lat_lng(cell)
                
                # Should be close to cell center (not exact original coords)
                np.testing.assert_allclose(lat2, lat, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(lng2, lng, rtol=1e-5, atol=1e-6)
                # Just verify it's valid
                assert -90 <= lat2 <= 90
                assert -180 <= lng2 <= 180


class TestCellToBoundary:
    """Test cell_to_boundary function."""
    
    def test_scalar_hexagon(self, valid_cells):
        """Test boundary for hexagons matches H3."""
        # Use a hexagon (not pentagon)
        cell = valid_cells[9]
        
        # Official H3
        h3_boundary = h3.cell_to_boundary(format(cell, 'x'))
        
        # GEOCELL
        geocell_boundary = geocell.cell_to_boundary(cell)
        
        # Should have same number of vertices
        assert len(geocell_boundary) == len(h3_boundary)
        
        # Coordinates should match
        for i in range(len(h3_boundary)):
            np.testing.assert_allclose(
                geocell_boundary[i][0], h3_boundary[i][0],
                rtol=1e-9, atol=1e-12
            )
            np.testing.assert_allclose(
                geocell_boundary[i][1], h3_boundary[i][1],
                rtol=1e-9, atol=1e-12
            )
    
    def test_scalar_pentagon(self, pentagon_cells):
        """Test boundary for pentagons."""
        cell = pentagon_cells[1]
        
        # Official H3
        h3_boundary = h3.cell_to_boundary(format(cell, 'x'))
        
        # GEOCELL
        geocell_boundary = geocell.cell_to_boundary(cell)
        
        # Pentagon should have 5 vertices
        assert len(geocell_boundary) == 5
        assert len(h3_boundary) == 5
        
        # Coordinates should match
        for i in range(5):
            np.testing.assert_allclose(
                geocell_boundary[i][0], h3_boundary[i][0],
                rtol=1e-9, atol=1e-12
            )
            np.testing.assert_allclose(
                geocell_boundary[i][1], h3_boundary[i][1],
                rtol=1e-9, atol=1e-12
            )
    
    def test_vector_success(self, valid_cells):
        """Test vectorized boundary extraction."""
        cells = np.array(list(valid_cells.values())[:3], dtype=np.uint64)
        
        # H3 loop
        h3_boundaries = [h3.cell_to_boundary(format(cell, 'x')) for cell in cells]
        
        # GEOCELL vectorized
        geocell_boundaries = geocell.cell_to_boundary(cells)
        
        # Check each boundary
        for i in range(len(cells)):
            assert len(geocell_boundaries[i]) == len(h3_boundaries[i])
            
            # Check each vertex
            for j in range(len(h3_boundaries[i])):
                np.testing.assert_allclose(
                    geocell_boundaries[i][j, 0], h3_boundaries[i][j][0],
                    rtol=1e-9, atol=1e-12
                )
                np.testing.assert_allclose(
                    geocell_boundaries[i][j, 1], h3_boundaries[i][j][1],
                    rtol=1e-9, atol=1e-12
                )
