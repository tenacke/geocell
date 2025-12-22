"""Type stubs for geocell._cython.h3.indexing (scalar functions)."""

import numpy as np
from numpy.typing import NDArray

def lat_lng_to_cell(lat: float, lng: float, res: int) -> int:
    """
    Convert latitude and longitude (in degrees) to an H3 cell index at the specified resolution.

    Parameters
    ----------
    lat : float
        Latitude in degrees
    lng : float
        Longitude in degrees
    res : int
        H3 resolution (0-15)

    Returns
    -------
    int
        H3 cell index

    Raises
    ------
    ValueError
        If latitude/longitude or resolution is invalid
    """
    ...

def cell_to_lat_lng(h3_index: int) -> tuple[float, float]:
    """
    Convert an H3 cell index to latitude and longitude (in degrees).

    Parameters
    ----------
    h3_index : int
        H3 cell index

    Returns
    -------
    tuple[float, float]
        (latitude, longitude) in degrees

    Raises
    ------
    ValueError
        If H3 index is invalid
    """
    ...

def cell_to_boundary(h3_index: int) -> NDArray[np.float64]:
    """
    Get the boundary vertices of an H3 cell index as a 2D array of latitude and longitude (in degrees).

    Parameters
    ----------
    h3_index : int
        H3 cell index

    Returns
    -------
    NDArray[np.float64]
        Array with shape (n_verts, 2) containing (lat, lng) pairs

    Raises
    ------
    ValueError
        If H3 index is invalid
    """
    ...
