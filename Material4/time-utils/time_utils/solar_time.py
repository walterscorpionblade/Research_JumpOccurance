from datetime import datetime, timedelta
from numpy import pi, cos, sin


def local_solar_time(dt, lon):
    '''Computes and returns local solar time for longitude `lon` given datetime `dt`'''
    gamma = 2 * pi / 365 * (dt.timetuple().tm_yday - 1 + float(dt.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma) \
             - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma))
    decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) \
           - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma) \
           - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
    time_offset = eqtime + 4 * lon
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    solar_time = datetime(dt.year, dt.month, dt.day) + timedelta(minutes=tst)
    return solar_time
