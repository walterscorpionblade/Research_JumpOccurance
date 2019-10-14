from numpy import ndarray, asarray
from datetime import datetime, timezone
from .gpst import gpst2dt

def time2jd(times):
    '''Return Julian day for given time or set of times.
    `times` is of form:
        ndarray -- each entry is interpreted as time in GPS seconds
        single float -- interpreted as GPS seconds
        datetime -- interpreted as UTC datetime
        list of datetime objects -- interpreted as UTC datetime
    See section 4-7: http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf
    ''' 
    if isinstance(times, float):  # convert single float to single datetime
        times = gpst2dt(times)
    if isinstance(times, datetime):  # is single time
        year, month = times.year, times.month
        days = (times - datetime(year, month, 1, tzinfo=timezone.utc)).total_seconds() / 86400
        return 367 * year - int(7 * int((month + 9) / 12) / 4) + int(275 * month / 9) + days
    if isinstance(times, ndarray):  # convert ndarray of GPS times to list of datetimes
        times = [gpst2dt(float(t)) for t in times]
    if isinstance(times, list):  # is multiple times
        year = asarray([dt.year for dt in times])
        month = asarray([dt.month for dt in times])
        days = asarray([(dt - datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)).total_seconds() / 86400 for dt in times])
        return 367 * year - (7 * ((month + 9) / 12).astype(int) / 4).astype(int) + (275 * month / 9).astype(int) + days

