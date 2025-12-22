"""Type stubs for geocell._cython.h3.indexing_vec (vectorized functions)."""

import numpy as np
from numpy.typing import NDArray

def lat_lng_to_cell_vec(
    lat: NDArray[np.float64], lng: NDArray[np.float64], res: int
) -> NDArray[np.uint64]:
    """
    Convert latitude and longitude (in degrees) arrays to an H3 cell index array at the specified resolution.

    Parameters
    ----------
    lat : NDArray[np.float64]
        Array of latitudes in degrees
    lng : NDArray[np.float64]
        Array of longitudes in degrees
    res : int
        H3 resolution (0-15)

    Returns
    -------
    NDArray[np.uint64]
        Array of H3 cell indices

    Raises
    ------
    ValueError
        If latitude and longitude arrays have different lengths
        If latitude/longitude or resolution is invalid
    """
    ...

def cell_to_lat_lng_vec(
    h3_indices: NDArray[np.uint64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert an array of H3 cell indices to latitude and longitude (in degrees) arrays.

    Parameters
    ----------
    h3_indices : NDArray[np.uint64]
        Array of H3 cell indices

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        (latitudes, longitudes) arrays in degrees

    Raises
    ------
    ValueError
        If any H3 index is invalid
    """
    ...

def cell_to_boundary_vec(h3_indices: NDArray[np.uint64]) -> NDArray[np.float64]:
    """
    Convert an array of H3 cell indices to their boundary vertices (latitude and longitude in degrees).

    Parameters
    ----------
    h3_indices : NDArray[np.uint64]
        Array of H3 cell indices

    Returns
    -------
    NDArray[np.float64]
        2D array with shape (n, 20) containing flattened (lat, lng) pairs
        Each row has up to 10 vertices stored as [lat0, lng0, lat1, lng1, ...]

    Raises
    ------
    ValueError
        If any H3 index is invalid
    """
    ...
