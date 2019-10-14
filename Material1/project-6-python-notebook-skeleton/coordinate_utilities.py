import numpy
from numpy import array, zeros, sin, cos, tan, arctan2, arcsin, sqrt, radians, degrees, column_stack


def ecef2geo(ecef):
    """Converts ECEF coordinates to geodetic coordinates, 

    Parameters
    ----------
    X : an array of N ECEF coordinates with shape (N,3).

    Returns
    -------
    output : (N,3) array
        geodetic coordinates in degrees and meters (lat, lon, alt)
    """
    ecef = ecef.reshape((-1, 3))
    rf = 298.257223563 # Earth reciprocal flattening (1/f)
    a = 6378137.       # Earth semi-major axis (m)
    b = a - a / rf     # Earth semi-minor axis derived from f = (a - b) / a
    x = ecef[:,0]
    y = ecef[:,1]
    z = ecef[:,2]

    # We must iteratively derive N
    lat = arctan2(z, sqrt(x**2 + y**2))
    h = z / sin(lat)
    d_h = 1.; d_lat = 1.

    N = zeros(ecef.shape[0])
    
    timeout = 200
    while (d_h > 1e-10) and (d_lat > 1e-10):
        N[:] = a**2 / (sqrt(a**2 * cos(lat)**2 + b**2 * sin(lat)**2))
        N1 = N * (b / a)**2
    
        temp_h = sqrt(x**2 + y**2) / cos(lat) - N
        temp_lat = arctan2(z / (N1 + h), sqrt(x**2 + y**2) / (N + h))
        d_h = max(abs(h - temp_h))
        d_lat = max(abs(lat - temp_lat))

        h = temp_h
        lat = temp_lat
        timeout -= 1
        if timeout <= 0:
            break

    lon = arctan2(y, x)
    lat = degrees(lat)
    lon = degrees(lon)

    geo = column_stack((lat, lon, h))
    return geo.reshape((-1, 3)).squeeze()


def geo2ecef(geo):
    """Converts geodetic coordinates to ECEF coordinates

    Parameters
    ----------
    geo : array of shape (N, 3)
        geodetic coordinates (lat, lon, alt)

    Returns
    -------
    output : array of shape (N, 3)
        ECEF coordinates
    """
    geo = geo.reshape((-1, 3))
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

    ecef = column_stack((x, y, z))
    return ecef.reshape((-1, 3)).squeeze()


def ecef2enu(ecef_ref, ecef_obj):
    """Converts ECEF coordinates to observer-relative ENU coordinates.

    Parameters
    ----------
    ecef_ref : array of shape (3,) or (1,3)
        observer ECEF coordinate
    ecef_obj : array of shape(N,3)
        object ECEF coordinates

    Returns
    -------
    output : array of shape(N,3)
        The east-north-up coordinates
    """
    ecef_ref = ecef_ref.reshape((-1, 3))
    ecef_obj = ecef_obj.reshape((-1, 3))
    # get the lat and lon of the user position
    geo = ecef2geo(ecef_ref)
    geo = geo.reshape((-1, 3))
    lat = radians(geo[:, 0])
    lon = radians(geo[:, 1])
    N = geo.shape[0]

    # create the rotation matrix
    Rl = array([[-sin(lon),                        cos(lon), zeros((N,))],
                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                [ cos(lat) * cos(lon),  cos(lat) * sin(lon), sin(lat)]])
    dx = ecef_obj - ecef_ref
    return numpy.sum(Rl * dx.T[None, :, :], axis=1).T  # sum across columns


def ecef2sky(ecef_ref, ecef_obj):
    """Converts observer and object ECEF coordinates to azimuth and elevation relative to observer.

    Parameters
    ----------
    ecef_ref : array of shape (3,) or (1,3)
        observer coordinate
    ecef_obj : array of shape(N,3)
        object coordinates

    Returns
    -------
    output : array of shape(N,2)
        The objects' sky coordinatescoordinates
        (azimuth, elevation in degrees)
    """
    enu = ecef2enu(ecef_ref, ecef_obj)
    enu = enu.reshape((-1, 3))
    e = enu[:, 0]
    n = enu[:, 1]
    u = enu[:, 2]
    az = arctan2(e, n)
    el = arcsin(u / sqrt(e**2 + n**2 + u**2))
    return column_stack((degrees(az), degrees(el))).squeeze()



