"""
coordinate_utils.py

@author Brian Breitsch
@email brianbreitsch@gmail.com
"""

from time_utils.gmst import time2gmst
from datetime import datetime, timedelta, timezone
import numpy
from numpy import sin, cos, tan, array, zeros, ones, sqrt, arctan2, radians, degrees, rad2deg, deg2rad, column_stack, absolute, arcsin, pi, ndarray

def make_3_tuple_array(arr):
    """Reshapes ndarray so that it has dimensions (N,3)
    """
    arr = array(arr) if isinstance(arr, list) or isinstance(arr, tuple) else arr
    assert(arr.ndim <= 2 and arr.size >= 3)
    if arr.shape[0] == 3:
        arr = arr.reshape((1,3)) if arr.ndim == 1 else arr.T
    assert(arr.shape[1] == 3)
    return arr


def dms2deg(arr):
    '''Converts degrees, minutes, seconds to degrees.
    Parameters
    ----------
    arr : list, tuple, or ndarray of shape (N,3) (or of length 3)
    The minutes/seconds should be unsigned but the degrees may be signed

    Returns
    -------
    (N, 3) array of degrees (or scalar in N == 1)
    '''
    arr = make_3_tuple_array(arr)
    return (numpy.sign(arr[:, 0]) * (abs(arr[:, 0]) + arr[:, 1] / 60.0 + arr[:, 2] / 3600.0)).squeeze()


def ecef2geo(x_ref, N=None, timeout=None):
    """Converts ECEF coordinates to geodetic coordinates, 

    Parameters
    ----------
    x_ref : an ndarray of N ECEF coordinates with shape (N,3).
    timeout : iterations will end when timeout goes to zero.

    Returns
    -------
    output : (N,3) ndarray
        geodetic coordinates in degrees and meters (lat, lon, alt)
    `N` : (optional) the radius of curvature, passed in as output parameter

    Notes
    -----
    >>> from numpy import array, radians
    >>> geo = array([27.174167, 78.042222, 0])  # lat, lon, alt
    >>> ecef = geo2ecef(radians(geo))
    >>> new_geo = ecef2geo(ecef)
    array([[             nan],
    	   [  7.08019709e+01],
           [ -6.37805436e+06]])
    >>> # [1176.45, 5554.887, 2895.397] << should be this 
    >>> ecef2geo(array([
    	[27.174167, 78.042222, 0],
    	[39.5075, -84.746667, 0]])).reshaps((3,2))
    array([[             nan,              nan],
           [  7.08019709e+01,  -6.50058423e+01],
           [ -6.37805436e+06,  -6.37804350e+06]])
    [1176.45, 5554.887, 2895.397]
    [451.176, -4906.978, 4035.946]
    """
    x_ref = make_3_tuple_array(x_ref)
    # we = 7292115*1**-11 # Earth angular velocity (rad/s)
    # c = 299792458    # Speed of light in vacuum (m/s)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    a = 6378137.       # Earth semi-major axis (m)
    b = a - a / rf     # Earth semi-minor axis derived from f = (a - b) / a
    if x_ref.shape == (3,):
        x_ref = x_ref.reshape((1,3))
    x = x_ref[:,0]; y = x_ref[:,1]; z = x_ref[:,2];

    # We must iteratively derive N
    lat = arctan2(z, sqrt(x**2 + y**2))
    h = z / sin(lat)
    d_h = 1.; d_lat = 1.

    if numpy.any(N) is None:
        N = zeros(x_ref.shape[0])
    
    while (d_h > 1e-10) and (d_lat > 1e-10):
        N[:] = a**2 / (sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2))
        N1 = N * (b / a)**2
    
        temp_h = sqrt(x**2 + y**2) / cos(lat) - N
        temp_lat = arctan2(z / (N1 + h), sqrt(x**2 + y**2) / (N + h))
        d_h = numpy.max(absolute(h - temp_h))
        d_lat = numpy.max(absolute(lat - temp_lat))

        h = temp_h
        lat = temp_lat
        if timeout is not None:
            timeout -= 1
            if timeout <= 0:
                break
    
    lon = arctan2(y,x)

    lat = degrees(lat)
    lon = degrees(lon)

    geo = column_stack((lat, lon, h))
    return geo.squeeze()


def local_enu(lat, lon):
    Rl = array([[-sin(lon),                        cos(lon), zeros((lat.shape[0],)) if isinstance(lat, numpy.ndarray) else 0],
                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                [ cos(lat) * cos(lon),  cos(lat) * sin(lon), sin(lat)]])
    return Rl


def ecef2enu(x_ref, x_obj, timeout=None):
    """Converts satellite ECEF coordinates to user-relative ENU coordinates.

    Parameters
    ----------
    x_ref : ndarray of shape (3,)
        observer ECEF coordinate
    x_obj : ndarray of shape(N,3)
        object ECEF coordinates
    timeout : int
        timeout integer, passed to ecef2geo

    Returns
    -------
    output : ndarray of shape(N,3)
        The east-north-up coordinates
    """
    x_ref = make_3_tuple_array(x_ref)
    x_obj = make_3_tuple_array(x_obj)
    # get the lat and lon of the user position
    geo = ecef2geo(x_ref, timeout=timeout)
    geo = make_3_tuple_array(geo)
    lat = radians(geo[:, 0])
    lon = radians(geo[:, 1])
    N = geo.shape[0]

    # create the rotation matrix
    Rl = local_enu(lat, lon)
    dx = x_obj - x_ref
    return numpy.sum(Rl * dx.T[None, :, :], axis=1).T  # sum across columns
    #return Rl.dot(dx.T).T.squeeze()

 
def enu2sky(enu):
    """Converts local East-North-Up coordinates to Sky coordinates (azimuth, elevation, radius)

    Parameters
    ----------
    enu : ndarray of shape(N,3)
        ENU coordinates

    Returns
    -------
    output : ndarray of shape(N,3)
        The sky coordinates
        (azimuth, elevation, radius)  in degrees and meters
    """
    enu = make_3_tuple_array(enu)
    e = enu[:, 0]; n = enu[:, 1]; u = enu[:, 2]
    az = arctan2(e, n)
    r = sqrt(e**2 + n**2 + u**2)
    el = arcsin(u / r)
    return column_stack((degrees(az), degrees(el), r)).squeeze()

def sky2enu(sky):
    """Converts local Sky coordinates back to local East-North-Up coordinates.
    """
    sky = make_3_tuple_array(sky)
    az, el, r = sky[:, 0], sky[:, 1], sky[:, 2]
    x = r * array([1, 0, 0])
    theta = radians(90 - az)
    phi = radians(el)
    e = r * cos(theta) * cos(phi)
    n = r * sin(theta) * cos(phi)
    u = r * sin(phi)
    return column_stack((e, n, u)).squeeze()


def ecef2sky(x_ref, x_obj, timeout=None):
    """Converts user and satellite ecef coordinates to azimuth and elevation
    from user on Earth, by first computing their relative ENU coordinates.  See `enu2sky`.

    Parameters
    ----------
    x_ref : ndarray of shape (3,)
        observer coordinate
    xs : ndarray of shape(N,3)
        object coordinates
    timeout: timout integer--passed to ecef2enu

    Returns
    -------
    output : ndarray of shape(N,3)
        The objects' sky coordinatescoordinates
        (azimuth, elevation, radius)  in degrees and meters
    """
    enu = ecef2enu(x_ref, x_obj, timeout=timeout)
    return enu2sky(enu)


def local_radius_of_curvature(geo):
    '''Returns local radius of curvature given geodetic coordinates.
    '''
    geo = make_3_tuple_array(geo)
    a = 6378137. # Earth semi-major axis (m)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf # Earth semi-minor axis derived from f = (a - b) / a
    lat = radians(geo[:, 0])
    lon = radians(geo[:, 1])
    h = geo[:, 2]

    N = a**2 / sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2)
    N1 = N * (b / a)**2
    return N1

   
def geo2ecef(geo):
    """Converts geodetic coordinates to ECEF coordinates

    Parameters
    ----------
    geo : ndarray of shape (N,3)
        geodetic coordinates (lat, lon, alt) in degrees and meters above WGS84 ellipsoid

    Returns
    -------
    output : ndarray of shape(N,3)
        ECEF coordinates

    Notes
    -----
    """
    geo = make_3_tuple_array(geo)
    a = 6378137. # Earth semi-major axis (m)
    rf = 298.257223563 # Reciprocal flattening (1/f)
    b = a * (rf - 1) / rf # Earth semi-minor axis derived from f = (a - b) / a
    lat = radians(geo[:, 0])
    lon = radians(geo[:, 1])
    h = geo[:, 2]

    N = a**2 / sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2)
    N1 = N * (b / a)**2

    x = (N + h) * cos(lat) * cos(lon)
    y = (N + h) * cos(lat) * sin(lon)
    z = (N1 + h) * sin(lat)

    x_ref = column_stack((x, y, z))
    return x_ref.squeeze()


def geo2sky(geo_ref, geo_obj):
    """Converts object geodetic coordinates to azimuth and elevation from
    reference geodetic coordinates on Earth by first computing their relative
    Sky coordinates.  See `enu2sky`.

    Parameters
    ----------
    geo_ref : ndarray of shape (3,)
        geodetic (lat, lon, alt) coordinates of observer
    geo_obj : ndarray of shape (N,3)
        geodetic (lat, lon, alt) coordinates of object

    Returns
    -------
    output : ndarray of shape(N,3)
        sky coordinates (azimuth, elevation, radius) in degrees and meters
    """
    geo_ref = make_3_tuple_array(geo_ref)
    geo_obj = make_3_tuple_array(geo_obj)
    x_ref = geo2ecef(geo_ref)
    x_obj = geo2ecef(geo_obj)
    return ecef2sky(x_ref, x_obj)


def eci2ecef(eci, time):
    """Converts eci coordinates to ecef coordinates given
    the time or times for said coordinates

    Parameters
    ----------
    eci : ndarray of shape (N,3)
    time : time array in GPS seconds of shape (N,)
        TODO fix ^^^
        could be any format accepted by `time2gmst`, which is declared in `time_utilities` package

    Returns
    -------
    output : ndarray of shape (N,3)
        ecef coordinates

    Notes
    Time of 2000 January 1, 12 UTC is (in GPS seconds) 630763213.0
    -----
    See: http://physics.stackexchange.com/questions/98466/radians-to-rotate-earth-to-match-eci-lat-lon-with-ecef-lat-lon

    TODO: only accepts one time right now
    """
    gmst = time2gmst(time)
    gmst = gmst % 24
    theta = 2 * pi / 24 * gmst
    N = len(theta)
    rot = array([[ cos(theta),  sin(theta), zeros((N,))],
                 [-sin(theta),  cos(theta), zeros((N,))],
                 [zeros((N,)), zeros((N,)), ones((N,))]])
    # rot is (3, 3, N), eci is (N, 3)
    return numpy.sum(rot * eci.T[None, :, :], axis=1).T  # sum across columns


def ecef2eci(ecef, time):
    """Converts ecef coordinates to eci coordinates given
    the time or times for said coordinates

    Parameters
    ----------
    ecef : ndarray of shape (N,3)
    time : any format accepted by `time2gmst`

    Returns
    -------
    output : ndarray of shape (N,3)
        eci coordinates

    Notes
    -----
    See: http://physics.stackexchange.com/questions/98466/radians-to-rotate-earth-to-match-eci-lat-lon-with-ecef-lat-lon
    Time of 2000 January 1, 12 UTC is (in GPS seconds) 630763213.0
    """
    gmst = time2gmst(time)
    gmst = gmst % 24
    theta = 2 * pi / 24 * gmst
    N = len(theta)
    # NOTE: super important to use array and not asarray for matrices
    rot = array([[ cos(theta), -sin(theta), zeros((N,))],
                 [ sin(theta),  cos(theta), zeros((N,))],
                 [zeros((N,)), zeros((N,)), ones((N,))]])
    return numpy.sum(rot * ecef.T[None, :, :], axis=1).T  # sum across columns
