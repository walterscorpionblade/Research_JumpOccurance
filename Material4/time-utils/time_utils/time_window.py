'''
TimeWindow class for iterating and checking time period overlap
'''

import datetime
from datetime import timedelta


class TimeWindow(object):
    '''Stores a datetime objects `start` and `end` and ensures `start < end` TODO'''
    
    def __init__(self, start, end):
        '''
        `start` -- beginning of time window
        `end` -- end of time window
        '''
        if type(start) is not datetime.datetime or type(end) is not datetime.datetime:
            raise TypeError('arg must be a datetime.datetime object')
        self.start = start
        self.end = end
    
    @staticmethod
    def plus_or_minus(date, hours=0, minutes=0, delta=None):
        '''Return a TimeWindow with start date `date - delta` and end date
        `date + delta` where `delta` is either specified directly by the user
        (i.e. uses `delta` parameter) or contructed from `hours` and `minutes`'''
        delta = delta if delta else datetime.timedelta(hours=hours, minutes=minutes)
        if type(date) is not datetime.datetime:
            raise TypeError('expected arguments are datetime.datetime and datetime.timedelta objects')
        return TimeWindow(date - delta, date + delta)

    def one_day_window(date):
        '''Return a TimeWindow that lasts from `date` to `date + datetime.timedelta(days=1)`'''
        return TimeWindow(date, date + datetime.timedelta(days=1))

    def one_year_window(date):
        '''Return a TimeWindow that lasts from `date` to `date.year += 1`'''
        one_year_from_date = datetime.datetime(date.year + 1, date.month, date.day, tzinfo=date.tzinfo)
        return TimeWindow(date, one_year_from_date)

    def year_window(year):
        '''Returns a window lasting 1 January of year to 31 December of year'''
        return TimeWindow(datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc), datetime.datetime(year, 12, 31, tzinfo=datetime.timezone.utc))

    def __contains__(self, date):
        return self.start <= date <= self.end

    def expand(self, delta):
        '''Returns a new TimeWindow with self.start - delta, self.end + delta'''
        return TimeWindow(self.start - delta, self.end + delta)

    def compress(self, delta):
        '''Returns a new TimeWindow with self.start + delta, self.end - delta'''
        new_start = self.start + delta
        new_end = self.end - delta
        if new_end < new_start:
            raise Exception('TimeWindow start cannot be > than end')
        return TimeWindow(new_start, new_end)

    def iterate(self, delta, offset=timedelta(seconds=0)):
        '''Iterates over time window by `delta`.'''
        dt = self.start + offset
        while dt < self.end:
            yield dt
            dt += delta

    @property
    def days(self):
        '''Iterates by days'''
        return self.iterate(datetime.timedelta(days=1))

    @property
    def duration(self):
        '''Returns timedelta which is end - start'''
        return self.end - self.start

    @property
    def middle(self):
        '''Returns datetime at middle of window'''
        return self.start + 0.5 * self.duration


