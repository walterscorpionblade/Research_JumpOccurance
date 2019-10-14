import numpy
from numpy import array, arange, pi, sin, cos, arctan2, sqrt, radians, degrees
from numpy.linalg import norm
from types import SimpleNamespace

def parse_yuma_almanacs(lines):
    '''Given the lines of a Yuma almanac file, produces a dictionary {sat_id: <namespace>},
    where `namespace` contains the almanac parameters.
    
    Input:
        `lines` -- list containing lines from a Yuma almanac file
    Output:
        `almanacs` -- dictionary {<sat_id>: <namespace>} where namespace contains:
            `Sat_ID` -- satellite ID or PRN
            `Health` -- satellite health indicator
            `Eccentricity` -- satellite orbit eccentricity
            `Time_of_Applicability` -- time of applicability of almanac ephemeris for satellite
            `Orbital_Inclination` -- satellite orbit inclination
            `Rate_of_RAAN` -- rate of change of RAAN in ECI frame
            `Sqrt_a` -- square root of orbit semi-major axis (m^(1/2))
            `RAAN_at_Week` -- RAAN at start of corresponding GPS week (rad.)
            `Arg_of_Perigee` -- orbit argument of perigee (rad.)
            `Mean_Anomaly` -- satellite mean anomaly (rad.)
            `Af0` -- coarse GPS clock correction bias term (s)
            `Af1` -- coarse GPS clock correction frequency term (s/s)
            `Week` -- GPS week number (mod 1024)
    '''
    index = 0
    L = len(lines)
    almanacs = {}
    while index < L:
        if lines[index].startswith('ID') and index < L - 13:
            alm = SimpleNamespace()
            alm.Sat_ID = int(lines[index][25:])
            alm.Health = int(lines[index + 1][25:])
            alm.Eccentricity = float(lines[index + 2][25:])
            alm.Time_of_Applicability = float(lines[index + 3][25:])
            alm.Orbital_Inclination = float(lines[index + 4][25:])
            alm.Rate_of_RAAN = float(lines[index + 5][25:])
            alm.Sqrt_a = float(lines[index + 6][25:])
            alm.RAAN_at_Week = float(lines[index + 7][25:])
            alm.Arg_of_Perigee = float(lines[index + 8][25:])
            alm.Mean_Anomaly = float(lines[index + 9][25:])
            alm.Af0 = float(lines[index + 10][25:])
            alm.Af1 = float(lines[index + 11][25:])
            alm.Week = int(lines[index + 12][25:])
            almanacs[alm.Sat_ID] = alm
            index += 13
        else:
            index += 1
    return almanacs


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
        if numpy.all(abs(E_new - E) < tol):
            return E_new
        E = E_new
    raise Exception('Max number of iterations exceeded ({0}) when solving for Eccentric Anomaly'.format(timeout))



def compute_ecef_position_from_gps_almanac(alm, t, rollover=1, mu_E=3.986005e14, Omega_E_dot=7.2921151467e-5):
    '''Given Yuma almanac for a particular SV and a scalar or an array of times,
    extracts the necessary Keplerian parameters and computes satellite position
    in ECEF coordinates.  Note that, because ECEF is a rotating coordinate frame,
    RAAN (denoted `Omega`) will depend on time `t`, as will the true anomaly `nu`.
    
    Input:
        `alm` -- almanac namespace output from `parse_yuma_almanacs`. This
            almanac corresponds to one satellite.
        `t` -- time array of shape (N,) of GPST seconds.
        `rollover` -- (optional) if almanac specifies week number modulo-1024,
            this rollover indicates how many multiples of 1024 to add to get the
            actual week number.
        `mu_E` -- (optional) Earth gravitational parameter (m^3/s^2)
        `Omega_E_dot` -- (optional) WGS Earth rotation rate
    Output:
        array of shape (N, 3) of satellite ECEF coordinates
    '''
    #! first extract the Keplerian elements `a`, `e`, `i`, `omega` from almanac
    a = alm.Sqrt_a**2
    e = alm.Eccentricity
    i = alm.Orbital_Inclination
    omega = alm.Arg_of_Perigee
    #! next compute the time of week `TOW` corresponding to the GPST input `t`
    #! and compute `Dt` -- the difference between `TOW` and the almanac
    #! ephemeris time of applicability
    T_oa = alm.Time_of_Applicability
    TOW = t - (alm.Week + rollover * 1024) * 7 * 24 * 3600
    Dt = TOW - T_oa
    #! next, compute `Omega` (the RAAN) in the ECEF coordinate frame
    Omega_0 = alm.RAAN_at_Week
    Omega_dot = alm.Rate_of_RAAN
    Omega = Omega_0 - TOW * Omega_E_dot + Dt * Omega_dot
    #! compute mean motion, mean anomaly, and eccentric anomaly
    n = sqrt(mu_E / a**3)           # mean motion
    M = alm.Mean_Anomaly + n * Dt   # mean anomaly
    E = solve_kepler(M, e)          # eccentric anomaly
    #! compute true anomaly, orbital radius
    nu = arctan2(sqrt(1. - e**2) * sin(E) / (1 - e * cos(E)), (cos(E) - e) / (1 - e * cos(E)))
    r = a * (1 - e * cos(E))
    #! compute preliminary (x, y) orbit coordinates (follow steps from lecture 14 slide 19)
    #! using so-called "argument of latitude" (`omega + nu`), which will account for the 
    #! rotation by `omega` that we would normally do to convert from perifocal to ECEF frame
    x_orb = r * cos(omega + nu)
    y_orb = r * sin(omega + nu)
    #! apply rotations `R3(Omega)R1(i)` to the (x, y) coordinates derived above
    x_ecef = x_orb * cos(Omega) - y_orb * sin(Omega) * cos(i)  # transform from orbital system to ECEF system
    y_ecef = x_orb * sin(Omega) + y_orb * cos(Omega) * cos(i)
    z_ecef = y_orb * sin(i)
    return array([x_ecef, y_ecef, z_ecef]).T


