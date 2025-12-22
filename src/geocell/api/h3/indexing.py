from ..._cython.h3 import indexing as scalar
from ..._cython.h3 import indexing_vec as vectorized

from typing import Union, Iterable, Tuple
import numpy as np

# ============================================================================
# INDEXING FUNCTIONS
# ============================================================================


def lat_lng_to_cell(
    lat: Union[float, Iterable, np.ndarray],
    lng: Union[float, Iterable, np.ndarray],
    resolution: Union[int, Iterable, np.ndarray],
) -> Union[int, np.ndarray]:
    """Convert lat/lng to H3 cell index.

    Automatically dispatches to scalar or vectorized backend based on input type.

    Parameters
    ----------
    lat : float or array_like
        Latitude(s) in degrees
    lng : float or array_like
        Longitude(s) in degrees
    resolution : int or array_like
        H3 resolution(s) (0-15)

    Returns
    -------
    int or np.ndarray[uint64]
        H3 cell index(es)

    Examples
    --------
    >>> # Scalar usage
    >>> lat_lng_to_cell(37.3615593, -122.0553238, 9)
    617700169958293503

    >>> # Array usage
    >>> lat_lng_to_cell([37.36, 40.71], [-122.06, -74.01], 9)
    array([617700169958293503, 617700440621039615], dtype=uint64)

    >>> # Mixed usage
    >>> lat_lng_to_cell(37.36, -122.06, [7, 8, 9])
    array([617700169983459327, 617700169991847935, 617700169958293503], dtype=uint64)
    """
    # Scalar path
    if all(np.isscalar(x) for x in [lat, lng, resolution]):
        return scalar.lat_lng_to_cell(float(lat), float(lng), int(resolution))

    # Vectorized path
    return vectorized.lat_lng_to_cell_vec(np.asarray(lat), np.asarray(lng), np.asarray(resolution))

def cell_to_lat_lng(
    cells: Union[int, Iterable, np.ndarray],
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """Convert H3 cell to lat/lng.

    Parameters
    ----------
    cells : int or array_like
        H3 cell index(es)

    Returns
    -------
    tuple[float, float] or tuple[np.ndarray, np.ndarray]
        (latitude, longitude) or (latitudes, longitudes) in degrees

    Examples
    --------
    >>> # Scalar
    >>> cell_to_lat_lng(617700169958293503)
    (37.36155934, -122.05532382)

    >>> # Array
    >>> cell_to_lat_lng([617700169958293503, 617700440621039615])
    (array([37.36155934, 40.71272811]), array([-122.05532382,  -74.00601521]))
    """
    if np.isscalar(cells):
        return scalar.cell_to_lat_lng(int(cells))

    return vectorized.cell_to_lat_lng_vec(np.asarray(cells))


def cell_to_boundary(cells: Union[int, Iterable, np.ndarray]) -> Union[list, list]:
    """Get cell boundary vertices.

    Parameters
    ----------
    cells : int or array_like
        H3 cell index(es)

    Returns
    -------
    list or list[np.ndarray]
        For scalar: list of (lat, lng) tuples
        For array: list of arrays with shape (n_verts, 2)

    Examples
    --------
    >>> # Scalar
    >>> boundary = cell_to_boundary(617700169958293503)
    >>> len(boundary)
    6

    >>> # Array
    >>> boundaries = cell_to_boundary([617700169958293503, 617700440621039615])
    >>> len(boundaries)
    2
    """
    if np.isscalar(cells):
        return scalar.cell_to_boundary(int(cells))

    return vectorized.cell_to_boundary_vec(np.asarray(cells))