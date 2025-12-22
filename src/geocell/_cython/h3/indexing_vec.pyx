import numpy as np
cimport numpy as np

from cython cimport nogil

from .h3_api cimport (
    H3int,
    H3Error,
    LatLng,
    latLngToCell,
    cellToLatLng,
    cellToBoundary,
    CellBoundary,
    radsToDegs,
    degsToRads
)


def lat_lng_to_cell_vec(np.ndarray[np.double_t, ndim=1] lat, np.ndarray[np.double_t, ndim=1] lng, int res) -> H3int[:]: 
    """
    Convert latitude and longitude (in degrees) arrays to an H3 cell index array at the specified resolution.
    """
    cdef Py_ssize_t n = lat.shape[0]
    if lng.shape[0] != n:
        raise ValueError("Latitude and longitude arrays must have the same length")

    cdef np.ndarray[H3int, ndim=1] h3_indices = np.empty(n, dtype=np.uint64)

    cdef Py_ssize_t i
    cdef LatLng g
    cdef H3int h
    cdef H3Error err

    for i in range(n):
        g.lat = degsToRads(lat[i])
        g.lng = degsToRads(lng[i])

        err = latLngToCell(&g, res, &h)
        if err != 0:
            raise ValueError("Invalid latitude/longitude or resolution at index {}".format(i))

        h3_indices[i] = h
   
    return h3_indices


def cell_to_lat_lng_vec(np.ndarray[H3int, ndim=1] h3_indices) -> (double[:], double[:]):
    """
    Convert an array of H3 cell indices to latitude and longitude (in degrees) arrays.
    """
    cdef Py_ssize_t n = h3_indices.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] lat = np.empty(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] lng = np.empty(n, dtype=np.double)

    cdef Py_ssize_t i
    cdef LatLng g
    cdef H3Error err

    for i in range(n):
        err = cellToLatLng(h3_indices[i], &g)
        if err != 0:
            raise ValueError("Invalid H3 index at index {}".format(i))

        lat[i] = radsToDegs(g.lat)
        lng[i] = radsToDegs(g.lng)

    return lat, lng


def cell_to_boundary_vec(np.ndarray[H3int, ndim=1] h3_indices) -> double[:, :]:
    """
    Convert an array of H3 cell indices to their boundary vertices (latitude and longitude in degrees).
    Returns a 2D array where each row corresponds to a cell and contains the lat/lng pairs of its vertices.
    """
    cdef Py_ssize_t n = h3_indices.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] boundaries = np.empty((n, 20), dtype=np.double)  # Assuming max 10 vertices (lat/lng pairs)

    cdef Py_ssize_t i, j
    cdef CellBoundary boundary
    cdef H3Error err

    for i in range(n):
        err = cellToBoundary(h3_indices[i], &boundary)
        if err != 0:
            raise ValueError("Invalid H3 index at index {}".format(i))

        for j in range(boundary.num_verts):
            boundaries[i, j * 2] = radsToDegs(boundary.verts[j].lat)
            boundaries[i, j * 2 + 1] = radsToDegs(boundary.verts[j].lng)

    return boundaries