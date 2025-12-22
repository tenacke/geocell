"""Type stubs for geocell.api.h3.indexing (public API)."""

from typing import Union, Iterable, Tuple, overload
import numpy as np
from numpy.typing import NDArray, ArrayLike

# Scalar overloads
@overload
def lat_lng_to_cell(
    lat: float,
    lng: float,
    resolution: int,
) -> int: ...
# Array overloads
@overload
def lat_lng_to_cell(
    lat: ArrayLike,
    lng: ArrayLike,
    resolution: ArrayLike,
) -> NDArray[np.uint64]: ...
def lat_lng_to_cell(
    lat: Union[float, ArrayLike],
    lng: Union[float, ArrayLike],
    resolution: Union[int, ArrayLike],
) -> Union[int, NDArray[np.uint64]]:
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
    int or NDArray[np.uint64]
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
    ...

# Scalar overload
@overload
def cell_to_lat_lng(cells: int) -> Tuple[float, float]: ...
# Array overload
@overload
def cell_to_lat_lng(
    cells: ArrayLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def cell_to_lat_lng(
    cells: Union[int, ArrayLike],
) -> Union[Tuple[float, float], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Convert H3 cell to lat/lng.

    Parameters
    ----------
    cells : int or array_like
        H3 cell index(es)

    Returns
    -------
    tuple[float, float] or tuple[NDArray[np.float64], NDArray[np.float64]]
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
    ...

# Scalar overload
@overload
def cell_to_boundary(cells: int) -> NDArray[np.float64]: ...
# Array overload
@overload
def cell_to_boundary(cells: ArrayLike) -> NDArray[np.float64]: ...
def cell_to_boundary(
    cells: Union[int, ArrayLike],
) -> Union[NDArray[np.float64], NDArray[np.float64]]:
    """Get cell boundary vertices.

    Parameters
    ----------
    cells : int or array_like
        H3 cell index(es)

    Returns
    -------
    NDArray[np.float64]
        For scalar: array with shape (n_verts, 2) containing (lat, lng) pairs
        For array: array with shape (n, 20) containing flattened boundary data

    Examples
    --------
    >>> # Scalar
    >>> boundary = cell_to_boundary(617700169958293503)
    >>> boundary.shape
    (6, 2)

    >>> # Array
    >>> boundaries = cell_to_boundary([617700169958293503, 617700440621039615])
    >>> boundaries.shape
    (2, 20)
    """
    ...
