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

    # Scalar path
    if all(np.isscalar(x) for x in [lat, lng, resolution]):
        return scalar.lat_lng_to_cell(float(lat), float(lng), int(resolution))

    if np.isscalar(resolution):
        # Broadcast scalar resolution to array
        resolution = np.full(np.shape(np.asarray(lat)), int(resolution), dtype=np.int_)
    # Vectorized path
    return vectorized.lat_lng_to_cell_vec(np.asarray(lat), np.asarray(lng), np.asarray(resolution))

def cell_to_lat_lng(
    cells: Union[int, Iterable, np.ndarray],
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
   
    if np.isscalar(cells):
        return scalar.cell_to_lat_lng(int(cells))

    return vectorized.cell_to_lat_lng_vec(np.asarray(cells))


def cell_to_boundary(cells: Union[int, Iterable, np.ndarray]) -> Union[list, list]:
 
    if np.isscalar(cells):
        return scalar.cell_to_boundary(int(cells))

    return vectorized.cell_to_boundary_vec(np.asarray(cells))