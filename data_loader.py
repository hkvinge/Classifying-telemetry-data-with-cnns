prefix = '/data3/darpa/tamu/'

import calcom
import numpy as np
ccd = calcom.io.CCDataSet(prefix+'tamu_expts_01-27.h5')

n_mice = len(ccd.data)

# _i = 0  # internal counter, if needed

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
                            which=which,mode,
                            mode=mode,
                            window=window, 
                            mouse_id=ccd.data[i].mouse_id.value
                        )
        time_chunks_all.append( time_chunks )
    #

    time_chunks = np.vstack(time_chunks_all)

    return time_chunks
#

