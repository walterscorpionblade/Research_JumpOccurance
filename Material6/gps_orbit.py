import numpy
from numpy import array, sin, cos, arctan2, sqrt

def solve_kepler(M, e, E0=0, tol=1e-12, timeout=200):
    '''Given mean anomaly `M` and orbital eccentricity `e`, uses Newton's method
    to iteratively solve Kepler's equations for the eccentric anomaly.
    Iteration stops when the magnitude between successive values of eccentric
    anomaly `E` is less than `tol`, or when `timeout` iterations have occurred in
    which case an exception is thrown.
    
    Input:
        `M` -- mean anomaly as scalar or array of shape (N,)
        `e` -- eccentricity (scalar)
        `E0` -- initial guess for `E` as scalar or array of shape (N,)
        `tol` -- desired tolerance for `abs(E[k+1]-E[k]) < tol`
        `timeout` -- number of iterations to attempt before throwing timeout error
    Output:
        `E` -- the eccentric anomaly as scalar or array of shape (N,)
    '''
    E = E0
    g = lambda E: E - e * sin(E) - M
    g_prime = lambda E: 1 - e * cos(E)
    for i in range(timeout):
        E_new = E - g(E) / g_prime(E)
        if numpy.max(abs(E_new - E)) < tol:
            return E_new
        E = E_new
    raise Exception('Max number of iterations exceeded ({0}) when solving for Eccentric Anomaly'.format(timeout))


def compute_gps_orbital_parameters_from_ephemeris(eph, t, rollover=0):
    '''
    Given GPS ephemerides for a particular SV and a scalar or an array of times,
    extracts the necessary Keplerian parameters, applies precise corrections,
    and returns the orbital parameters necessary to compute satellite position
    in ECEF coordinates.  Note that, because ECEF is a rotating coordinate
    frame, LAN (denoted `Omega`) will depend on time `t`, as will the true
    anomaly `nu`.
    
    The first three parameters (semi-major axis, eccentricity,
    inclination) are normal Keplerian elements.
    
    The fourth parameter is the longitude of ascending node `Omega`, which
    includes Earth rotation so that the an orbit estimated using these
    parameters will be in the Earth-centered fixed coordinate system (ECEF).
    
    The last parameter is the argument of latitude `Phi` which accounts for
    perigee plus true anomaly.
    ----------------------------------------------------------------------------
    Input:
        `eph` -- ephemeris namespace output from `parse_rinex_nav_file`.
        `t` -- time array of shape (N,) of GPST seconds.
        `rollover` -- (optional) if ephemeris specifies week number modulo-1024,
            this rollover indicates how many multiples of 1024 to add to get the 
            actual week number.  Ephemeris usually provides full week number.
    Output:
        Namespace containing arrays of shape (N,) -- the satellite orbital
        parameters `a`, `e`, `i`, `r`, `Omega`, `Phi`, `n` `E`, which are
        necessary for ECEF position or velocity calculation.  These parameters
        will be self-consistent.
            a - orbitsemi-major axis
            e - orbital eccentricity
            i - orbital inclination
            r - orbital radius
            Omega - longitude of ascending node
            Phi - argument of latitude
            n - mean motion
            E - eccentric anomaly
    '''
    pi = 3.1415926535898
    mu_E = 3.986005e14
    Omega_E_dot = 7.2921151467e-5
    # first extract the Keplerian elements `a`, `e`, `i_0`/`i_dot`, `omega` from
    # ephemeris 
    a = eph['sqrt_a']**2
    e = eph['e']
    i_0 = eph['i_0']
    i_dot = eph['i_dot']
    omega = eph['omega']
    M_0 = eph['m_0']
    # next compute the time of week `TOW` corresponding to the GPST input `t`
    # and compute `Dt` -- the difference between `TOW` and the time of ephemeris
    T_oe = eph['t_oe']
    TOW = t - (eph['week'] + rollover * 1024) * 7 * 24 * 3600
    Dt = TOW - T_oe
    # next, compute `Omega` -- the so-called longitude of ascending node, or the RAAN
    # in the ECEF coordinate frame
    Omega_0 = eph['omega_0']
    Omega_dot = eph['omega_dot']
    Omega = Omega_0 - TOW * Omega_E_dot + Dt * Omega_dot  # longitude of ascending node
    # compute mean motion, mean anomaly, and eccentric anomaly
    n0 = sqrt(mu_E / a**3)          # intitial mean motion
    n = n0 + eph['delta_n']         # corrected mean motion
    M = M_0 + n * Dt                # mean anomaly
    E = solve_kepler(M, e)          # eccentric anomaly
    # compute true anomaly
    nu = arctan2(sqrt(1. - e**2) * sin(E), cos(E) - e)  # true anomaly
    # compute argument of latitude, then compute the secord-order harmonic perturbation
    # correction terms for argument of latitude, radius, and orbital inclination
    Phi = nu + omega  # argument of latitude
    du = eph['c_us'] * sin(2 * Phi) + eph['c_uc'] * cos(2 * Phi)
    dr = eph['c_rs'] * sin(2 * Phi) + eph['c_rc'] * cos(2 * Phi)
    di = eph['c_is'] * sin(2 * Phi) + eph['c_ic'] * cos(2 * Phi)
    # correct arg. of latitude, orbital radius, and inclination
    Phi = Phi + du  # argument of latitude (corrected)
    r = a * (1. - e * cos(E)) + dr  # orbital radius (corrected)
    i = i_0 + di + i_dot * Dt  # inclination (corrected)
    return {'a': a, 'e': e, 'i': i, 'r': r, 'Omega': Omega, 'Phi': Phi, 'n': n, 'E': E}

def compute_ecef_position_from_orbital_parameters(params):
    '''
    Computes and returns the satellite position given a set of orbital parameters
    that define the satellite position
    ----------------------------------------------------------------------------
    
    `params` -- namespace containing necessary orbital parameters:
        i - orbital inclination
        r - orbital radius
        Omega - longitude of ascending node
        Phi - argument of latitude
    '''
    #a, e, i, r, Omega, Phi, n, E = params.a, params.e, params.i, params.r, params.Omega, params.Phi, params.n, params.E
    i, r, Omega, Phi = params['i'], params['r'], params['Omega'], params['Phi']
    x_orb, y_orb = r * cos(Phi), r * sin(Phi)  # compute x, y in orbital plane
    x_ecef = x_orb * cos(Omega) - y_orb * sin(Omega) * cos(i)  # transform from orbital system to ECEF system
    y_ecef = x_orb * sin(Omega) + y_orb * cos(Omega) * cos(i)
    z_ecef = y_orb * sin(i)
    return array([x_ecef, y_ecef, z_ecef])


def compute_ecef_velocity_from_orbital_parameters(a, e, i, Omega, n, E):
    '''
    Computes and returns the satellite velocity given an appropriate set of
    orbital parameters.
    ----------------------------------------------------------------------------
    Input:
        `params` -- namespace containing appropriate orbital parameters; see
            e.g. `compute_gps_orbital_parameters_from_ephemeris`
            These are:
                e - orbital eccentricity
                i - orbital inclination
                Omega - longitude of ascending node
                n - mean motion
                E - eccentric anomaly
    '''
    e, i, Omega, n, E = params['e'], params['i'], params['Omega'], params['n'], params['E']
    v_x_orb = n * a * sin(E) / (1. - e * cos(E))
    v_y_orb = -n * a * sqrt(1. - e**2) * cos(E) / (1. - e * cos(E))
    v_x_ecef = v_x_orb * cos(Omega) - v_y_orb * sin(Omega) * cos(i)  # transform from orbital system to ECEF system
    v_y_ecef = v_x_orb * sin(Omega) + v_y_orb * cos(Omega) * cos(i)
    v_z_ecef = v_y_orb * sin(i)
    return array([v_x_ecef, v_y_ecef, v_z_ecef])
