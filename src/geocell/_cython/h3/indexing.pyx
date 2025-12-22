import numpy as np
cimport numpy as np

cimport h3_api
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

cpdef H3int lat_lng_to_cell(double lat, double lng, int res): 
    """
    Convert latitude and longitude (in degrees) to an H3 cell index at the specified resolution.
    """
    cdef LatLng g
    cdef H3int h
    cdef H3Error err

    g.lat = degsToRads(lat)
    g.lng = degsToRads(lng)

    err = latLngToCell(&g, res, &h)
    if err != 0:
        raise ValueError("Invalid latitude/longitude or resolution")

    return h


cpdef (double, double) cell_to_lat_lng(H3int h3_index):
    """
    Convert an H3 cell index to latitude and longitude (in degrees).
    """
    cdef LatLng g
    cdef H3Error err

    err = cellToLatLng(h3_index, &g)
    if err != 0:
        raise ValueError("Invalid H3 index")

    return radsToDegs(g.lat), radsToDegs(g.lng)


cpdef double[:, :] cell_to_boundary(H3int h3_index):
    """
    Get the boundary vertices of an H3 cell index as a 2D array of latitude and longitude (in degrees).
    """
    cdef CellBoundary boundary
    cdef H3Error err

    err = cellToBoundary(h3_index, &boundary)
    if err != 0:
        raise ValueError("Invalid H3 index")

    cdef int num_verts = boundary.num_verts
    cdef np.ndarray[np.double_t, ndim=2] verts = np.empty((num_verts, 2), dtype=np.double)

    cdef int i
    for i in range(num_verts):
        verts[i, 0] = radsToDegs(boundary.verts[i].lat)
        verts[i, 1] = radsToDegs(boundary.verts[i].lng)

    return verts
