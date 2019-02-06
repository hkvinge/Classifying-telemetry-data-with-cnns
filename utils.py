
import numpy as np

import calcom
prefix = '/data3/darpa/tamu/'
ccd = calcom.io.CCDataSet(prefix+'tamu_expts_01-27.h5')
n_mice = len(ccd.data)

def load_one(which='t', mode='chunk', window=1440, mouse_id=''):
    '''
    Loads a timeseries.

    Inputs:
        which : String; 't' for temperature, 'a' for activity (default: 't')
        mode : String; 'streaming' or 'chunk'. If 'streaming', a continuous sliding 
            window is run across the data; with the windows overlapping.
            If 'chunk', the windows do not overlap. (default: 'chunk')
        window : Integer; the size of the windows to be used in terms of 
                array size. The underlying units are in minutes. (default: 1440, or one day)
        mouse_id : String; reference to a specific mouse. (default: empty string)
        
    Outputs:
        time_chunks : numpy array of dimension n-by-d, where d is the window 
            size, and n is the number of timeseries segments salvageable from 
            the given mouse. This varies a lot depending on the mouse.

    Notes:
        If mouse_id is not specified, the global parameter _i is referenced 
        and the operations are applied on ccd.data[_i].
    '''

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

    if mode=='chunk':
        time_chunks = [ process_timeseries( tseries[ i*window : (i+1)*window ] ) for i in range(len_ts//window) ]
    elif mode=='streaming':
        pass
    else:
        raise ValueError('mode %s not recognized.'%str(mode))
    #

    time_chunks = np.vstack(time_chunks)

    return time_chunks
#

def load_all(which='t', mode='chunk', window=1440):
    '''
    Loads all timeseries by repeatedly calling load_one iteratively.

    Inputs:
        Same optional inputs as load_one, except mouse_id, with same default parameters.
    Outputs:
        time_chunks : numpy array of dimension N-by-d, where the arrays 
            from each result are concatenated.
    '''
    
    time_chunks_all = []

    for i in range(n_mice):
        time_chunks = load_one(
                            which=which,
                            mode=mode,
                            window=window, 
                            mouse_id=ccd.data[i].mouse_id.value
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
        verbosity: integer; 0 indicates no output. (default: 0)
    Outputs:
        tseries: a numpy array shape either (d,) or (0,d), depending on 
            whether the timeseries was thrown out for having too much 
            missing data.
    '''

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
