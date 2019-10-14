from types import SimpleNamespace
import numpy
from numpy import array
from datetime import datetime

def parse_header(lines):
    '''
    Given list of lines corresponding to the header of a RINEX observation file, parses
    the header of the file and returns a namespace containing the header information.
    
    Input
    -----
    `lines` -- lines corresponding to RINEX header
    
    Output
    ------
    namespace containing RINEX header information
    '''
    header = SimpleNamespace()
    for i, line in enumerate(lines):
        if line.find('# / TYPES OF OBSERV') >= 0:
            header.n_obs = int(line[:10])
            header.obs_types = line[10:58].split()
    return header

def parse_obs(lines, observations, century=2000):
    '''
    Given `lines` corresponding to the RINEX observation file data (non-header) lines,
    and a list of the types of observations recorded at each epoch, produces a dictionary
    containing the observation time and values for each satellite.
    
    Input
    -----
    `lines` -- data lines from RINEX observation file
    `observations` -- list of the observations reported at each epoch
    
    Output
    ------
    dictionary of format:
        {<sat_id>: {'time': [<dt...>], <obs_id>: [<values...>]}}
    '''
    data = {}  # <sat_id>: {'time': [<dt...>], <obs_id>: [<values...>]}
    lines = iter(lines)
    try:
        while True:
            # at each epoch, the two-digit year, month, day, hour, minute, and seconds
            # of the measurement epoch are specified, along with the number and ids of
            # the satellites whose measurements are given
            line = next(lines)
            yy = int(line[:4])
            year = century + yy
            month = int(line[4:7])
            day = int(line[7:10])
            hour = int(line[10:13])
            minute = int(line[13:16])
            seconds = float(line[16:25])
            microseconds = int(1e6 * (seconds % 1))
            seconds = int(seconds)
            dt = numpy.datetime64(datetime(year, month, day, hour, minute, seconds, microseconds))
            flag = int(line[25:28])
            num_sats = int(line[29:32])
            # there is space for (80 - 32) / 3 = 16 satellite ids
            # if there are more than 16, then they continue on the next line
            line = line[32:]
            if num_sats > 16:
                line = (line + next(lines).strip()).replace(' ', '')
            line = line.strip()
            # must replace spaces with zeros: e.g. to convert `'G 1'` to `'G01'`
            sat_ids = [line[3*i:3*(i+1)].replace(' ', '0') for i in range(num_sats)]

            for sat_id in sat_ids:
                # create new entry if `sat_id` is new
                if sat_id not in data.keys():
                    data[sat_id] = {'time': []}
                    for obs_id in observations:
                        data[sat_id][obs_id] = []
                # append time first, then append obs values
                data[sat_id]['time'].append(dt)
                # each line of observation values contains up to 5 entries
                # each entry is of width 16, starting at index 0
                num_lines_per_sat = 1 + len(observations) // 5
                line = ''
                while num_lines_per_sat > 0:
                    line += next(lines).replace('\n', '')
                    num_lines_per_sat -= 1
                for i in range(len(observations)):
                    val = float(line[16 * i:16 * (i + 1)])
                for i, obs_id in enumerate(observations):
                    try:
                        val = float(line[16 * i:16 * (i + 1)])
                    except Exception:
                        val = nan
                    data[sat_id][obs_id].append(val)
    except StopIteration:
        pass
    return data

def transform_values_from_rinex_obs(rinex_data):
    '''
    Transforms output from `parse_obs` to more useful format.
    
    Input:
    -------
    `rinex_data` -- Python dictionary with format:
        {<sat_id>: {'time': [<dt...>], <obs_id>: [<values...>]}}
        
    Output:
    -------
    `data` -- namespace containing:
        `satellites` -- dictionary of format {<sat_id>: <namespace>} with
        <namespace> containing time array and signal namespaces.  Each 
        signal namespace contains arrays of any measurements for that 
        corresponding signal.
    '''
    rinex_obs_datatypes_mapping = {
        'C1': {'signal': 'L1', 'name': 'pr'},
        'L1': {'signal': 'L1', 'name': 'carrier'},
        'D1': {'signal': 'L1', 'name': 'doppler'},
        'S1': {'signal': 'L1', 'name': 'snr'},
        'C2': {'signal': 'L2', 'name': 'pr'},
        'L2': {'signal': 'L2', 'name': 'carrier'},
        'D2': {'signal': 'L2', 'name': 'doppler'},
        'S2': {'signal': 'L2', 'name': 'snr'},
    }
    data = {}
    for sat_id, rnx_sat in rinex_data.items():
        if sat_id not in data.keys():
            data[sat_id] = SimpleNamespace(signals={})
        sat = data[sat_id]
        for obs_name, mapping in rinex_obs_datatypes_mapping.items():
            if obs_name in rnx_sat.keys():
                signal = mapping['signal']
                if signal not in sat.signals.keys():
                    sat.signals[signal] = SimpleNamespace()
                setattr(sat.signals[signal], mapping['name'], array(rnx_sat[obs_name]))
        if 'time' in rnx_sat.keys():
            sat.time = array(rnx_sat['time'])
    return data

def parse_rinex_obs_file(filepath):
    '''Given the filepath to a RINEX observation file, parses and returns header
    and observation data.
    
    Input
    -----
    `filepath` -- filepath to RINEX observation file
    
    Output
    ------
    `header, obs_data` where `header` is a namespace containing the parsed header information
        and `obs_data` is a namespace containing the observation data in the format:
        {<sat_id>: namespace(time=ndarray, signals={<sig_id>: namespace(<obs_name>=ndarray)})}
        
        Note: `time` on the satellite namespace is a `numpy.datetime64` object
    '''
    with open(filepath, 'r') as f:
        lines = list(f.readlines())
    for i, line in enumerate(lines):
        if line.find('END OF HEADER') >= 0:
            break
    header_lines = lines[:i + 1]
    obs_lines = lines[i + 1:]
    header = parse_header(header_lines)
    obs_data = parse_obs(obs_lines, header.obs_types)
    obs_data = transform_values_from_rinex_obs(obs_data)
    return header, obs_data


