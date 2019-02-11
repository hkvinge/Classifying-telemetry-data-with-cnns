
import numpy as np

import calcom
prefix = '/Users/HK/Programming/Calcom/tamu/'
ccd = calcom.io.CCDataSet(prefix+'tamu_expts_01-27.h5')
n_mice = len(ccd.data)

# Global variables; updated on calls to process_timeseries()
# To be used for get_labels().
global _mids
global _t_lefts

_mids = []
_t_lefts = []

def load_one(which='t', window=1440, step=np.nan, mouse_id='', reset_globals=True, nan_thresh=120):
    '''
    Loads a timeseries.

    Inputs:
        which : String; 't' for temperature, 'a' for activity (default: 't')
        step : Integer or nan-valued. This indicates the step taken in an individual
            timeseries before a new sample of size window is taken. If np.nan,
            then this defaults to the same value as window (i.e., timeseries are non-overlapping).
            (default: np.nan)
        window : Integer; the size of the windows to be used in terms of
                array size. The underlying units are in minutes. (default: 1440, or one day)
        mouse_id : String; reference to a specific mouse. (default: empty string)
        reset_globals : Boolean; whether to reset the global variables
            _mids and _t_lefts when this is being called. If this is being
            called on its own, you probably want this to be True.
            When this is called by load_all(), it is set False.
            (default: True)
        nan_thresh : integer. Number of NaN values in a chunk of the timeseries that 
            can be tolerated before the chunk is thrown out. (default: 120). 

    Outputs:
        time_chunks : numpy array of dimension n-by-d, where d is the window
            size, and n is the number of timeseries segments salvageable from
            the given mouse. This varies a lot depending on the mouse.

    Notes:
        - If mouse_id is not specified, the global parameter _i is referenced
        and the operations are applied on ccd.data[_i].
        - If the window size is smaller than nan_thresh, then nan_thresh will be 
        internally reset to window-1.
    '''
    import numpy as np

    nan_thresh = min(nan_thresh,window-1)

    # get index pointing to appropriate datatype
    datatype = {'t':0, 'a':1}[which]
    if ( not isinstance(mouse_id,str) ) or ( len(mouse_id)==0 ):
        print('Warning: invalid mouse_id specified. Returning empty array.')
        return np.zeros( (0,window) )
    #

    mid = ccd.find('mouse_id',mouse_id)[0]
    dp = ccd.data[mid]
    tseries = dp[datatype]

    len_ts = len(tseries)

    if np.isnan(step):
        step = window

    nchunks = ( len(tseries) - window )//step +1

    time_chunks = [ 
                    process_timeseries( tseries[ i*step : i*step + window ], nan_thresh=nan_thresh) 
                    for i in range(nchunks) 
                ]

    t_lefts_local = np.array([i*step for i in range(nchunks)])
    mouse_pointers_local = np.array([ mid for _ in range(nchunks) ])

    # update the globals based on what time_chunks looks like.
    shapes = np.array([np.prod( np.shape(tc) ) for tc in time_chunks])
    valid_tc = np.where(shapes!=0)[0]

    t_lefts_local = t_lefts_local[valid_tc]
    mouse_pointers_local = mouse_pointers_local[valid_tc]

    global _mids
    global _t_lefts

    if reset_globals:
        _mids = mouse_pointers_local
        _t_lefts = t_lefts_local
    else:
        _mids += list( mouse_pointers_local )
        _t_lefts += list( t_lefts_local )
    #

    time_chunks = np.vstack(time_chunks)

    return time_chunks
#

def load_all(which='t', window=1440, step=np.nan, nan_thresh=120):
    '''
    Loads all timeseries by repeatedly calling load_one iteratively.

    Inputs:
        Same optional inputs as load_one, except mouse_id, with same default parameters.
    Outputs:
        time_chunks : numpy array of dimension N-by-d, where the arrays
            from each result are concatenated.
    '''
    import numpy as np

    # Reset the global variables
    global _mids
    global _t_lefts
    _mids = []
    _t_lefts = []

    time_chunks_all = []

    for i in range(n_mice):
        time_chunks = load_one(
                            which = which,
                            window = window,
                            step = step,
                            mouse_id = ccd.data[i].mouse_id.value,
                            nan_thresh = nan_thresh,
                            reset_globals = False                            
                        )
        time_chunks_all.append( time_chunks )
    #

    time_chunks = np.vstack(time_chunks_all)

    return time_chunks
#

def process_timeseries(tseries_raw, nan_thresh=120,**kwargs):
    '''
    Applies preprocessing to a timeseries based on the rules:

    1. If the timeseries has missing data above a threshold,
        the datapoint is thrown out in the sense that
        an empty array of commensurate second dimension
        is returned.
    2. Else, NaNs in the data are filled by linear interpolation
        of neighboring data.

    Inputs:
        tseries_raw: a numpy array shape (d,), possibly containing NaNs.
    Optional inputs:
        nan_thresh: integer. If there are at least this many nans, nothing
            is done; instead an empty array of shape (0,d) is returned. (default: 120)
        verbosity: integer; 0 indicates no output. (default: 0)
    Outputs:
        tseries: a numpy array shape either (d,) or (0,d), depending on
            whether the timeseries was thrown out for having too much
            missing data.
    
    '''
    import numpy as np

    verbosity = kwargs.get('verbosity',0)

    d = len(tseries_raw)
    nnans = np.where(tseries_raw!=tseries_raw)[0]

    if len(nnans) >= nan_thresh:
        if verbosity!=0:
            print('The timeseries has greater than %i NaNs; throwing it out.'%nan_thresh)
        return np.zeros( (0,d) )
    #

    # Else, make a copy of the timeseries and clean it up in-place.
    tseries = np.array(tseries_raw)

    # Check beginning and end of timeseries for nans; replace
    # with the first non-nan value in the appropriate direction.
    if np.isnan( tseries[0] ):
        for i in range(nan_thresh):
            if not np.isnan(tseries[i]):
                break
        #
        tseries[:i] = tseries[i]
    #
    if np.isnan( tseries[-1] ):
        for i in range(d-1, d-nan_thresh-1, -1):
            if not np.isnan(tseries[i]):
                break
        #
        tseries[i:] = tseries[i]
    #

    nan_loc = np.where(np.isnan(tseries))[0]
    if len(nan_loc)==0:
        # Nothing to be done.
        return tseries
    #

    # Now work in the interior. Identify locations of
    # nans, identify all contiguous chunks, and fill them
    # in one by one with the nearest non-nan values.
    contig = []
    count = 0
    while count < len(nan_loc):
        active = nan_loc[count]
        # scan the following entries looking for a continguous
        # chunk of nan.
        chunk = [nan_loc[count]]
        for j in range(count+1,len(nan_loc)):
            if (nan_loc[j] == nan_loc[j-1] + 1):
                chunk.append( nan_loc[j] )
                count += 1
            else:
                break
        #

        # Identify the tether points for the linear interpolation.
        left = max(chunk[0] - 1, 0)
        fl = tseries[left]

        right = min(chunk[-1] + 1, len(tseries)-1)
        fr = tseries[right]

        m = (fr-fl)/(len(chunk) + 1)
        tseries[chunk] = fl + m*np.arange(1 , len(chunk)+1)

        count += 1
    #

    return tseries
#

def get_labels(attr):
    '''
    Returns an array of labels for the given attribute,
    AFTER a call to load_all() has been made. Running this
    prior to that will return an error. The reason is that
    the way the data is generated dynamically in
    load_one() and load_all() depends on the window and step
    parameters; global variables in utils are updated to
    keep track of information necessary specific to that
    partitioning of the data.

    Inputs:
        attr: string. Name of a built-in attribute in the dataset
            (see utils.ccd.attrnames) or one of a small set of
            additional attributes derived from these which are
            of interested to us (e.g. t_translated, t_since_infection).
    Outputs:
        labels: array of appropriate datatype for the requested information.
            The size will be commensurate to the data generated with
            the most recent call to utils.load_all(); e.g. if
            you have a data array of shape (12345,1440) then
            the labels will be shape (12345,).

    The list of additional attributes (as of February 7, 2019) are:
        t_translated : integer; (time since start of experiment) - (time of infection).
            The time since start of experiment is measured
            on the left endpoint of the chunk.
        t_post_infection : integer; max(0, t_translated). This presumes
            all pre-infection data is of the same quality.
        infected : 0/1 integer. This asks if the chunk is at a point 
            pre- or post-infection. Essentially (t_translated > 0) casted to integer.
    '''
    import numpy as np

    global _mids
    global _t_lefts

    if attr in ccd.attrnames:
        labels = ccd.get_attr_values(attr, idx=_mids)

    elif attr in ['t_translated', 't_post_infection', 'infected']:

        itimes = ccd.get_attr_values('infection_time', idx=_mids)
        t_translated = np.array(_t_lefts) - itimes

        if attr=='t_translated':
            labels = t_translated
        elif attr=='t_post_infection':
            labels = np.maximum(0., t_translated)
        elif attr=='infected':
            labels = np.array( t_translated > 0., dtype=int)
        #
    else:
        raise ValueError('Attribute %s not recognized. See the docstring for a list of allowable inputs.'%str(attr))
    #

    return labels
#

if __name__=="__main__":
    # Temporary testing
    from matplotlib import pyplot

    npoints = 50

    tseries_bad = np.random.choice(np.arange(-5.,6.), npoints)
    tseries_bad[4] = np.nan         # single point in the middle
    tseries_bad[-5:] = np.nan       # range at the tail of the timeseries
    tseries_bad[:2] = np.nan        # range at the head of the timeseries
    tseries_bad[7:10] = np.nan      # small range in the middle
    tseries_bad[11:21] = np.nan     # wider range in the middle

    tseries = process_timeseries(tseries_bad)

    fig,ax = pyplot.subplots(1,1)

    ax.plot( np.arange(npoints), tseries, c='r', marker='.', markersize=10, label='interpolated')
    ax.scatter( np.arange(npoints), tseries_bad, marker='o', edgecolor='k', s=100, label='original data')

    ax.legend(loc='upper right')
    fig.show()
#
