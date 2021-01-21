
from numpy import ones, zeros, sum

#  The following data is copied over from pages 6-7 of the GPS ICD IS-GPS-200H
#  A copy should be available here:  http://www.gps.gov/technical/icwg/IS-GPS-200H.pdf
# PRN : CA Phase Select
L1_CA_PHASE_SELECTS = {
    1 : (2, 6), 2 : (3, 7), 3 : (4, 8), 4 : (5, 9), 5 : (1, 9),
    6 : (2, 10), 7 : (1, 8), 8 : (2, 9), 9 : (3, 10), 10 : (2, 3),
    11 : (3, 4), 12 : (5, 6), 13 : (6, 7), 14 : (7, 8), 15 : (8, 9),
    16 : (9, 10), 17 : (1, 4), 18 : (2, 5), 19 : (3, 6), 20 : (4, 7),
    21 : (5, 8), 22 : (6, 9), 23 : (1, 3), 24 : (4, 6), 25 : (5, 7),
    26 : (6, 8), 27 : (7, 9), 28 : (8, 10), 29 : (1, 6), 30 : (2, 7),
    31 : (3, 8), 32 : (4, 9), 33 : (5, 10), 34 : (4, 10), 35 : (1, 7),
    36 : (2, 8), 37 : (4, 10),
}


def generate_mls(N, feedback_taps, output_taps):
    '''
    Generates maximum-length sequence (MLS) for the given linear feedback
    shift register (LFSR) length, feedback taps, and output taps.  The initial
    state of the LFSR is taken to be all ones.
    
    Parameters
    ----------------------------------------------------------------------------
    N : int
        length of LFSR
    feedback_taps : array or ndarray of shape (L,)
        the L taps to use for feedback to the shift register's first value
    output_taps : array or ndarray of shape (M,)
        the M taps to use for choosing the sequence output
    
    Returns
    ----------------------------------------------------------------------------
    output : ndarray of shape (2**N - 1,)
        the binary MLS values
    ''' 
    shift_register = ones((N,))
    values = zeros((2**N - 1,))
    for i in range(2**N - 1):
        values[i] = sum(shift_register[output_taps]) % 2 
        first = sum(shift_register[feedback_taps]) % 2 
        shift_register[1:] = shift_register[:-1]
        shift_register[0] = first
    return values


def generate_GPS_L1CA_code(prn):
    '''
    Generates GPS L1 C/A code for given PRN.
    
    Parameters
    ----------------------------------------------------------------------------
    prn : int 
        the signal PRN
    
    Returns
    ----------------------------------------------------------------------------
    output : ndarray of shape(1023,)
        the complete code sequence
    '''
    ps = L1_CA_PHASE_SELECTS[prn]
    g1 = generate_mls(10, [2, 9], [9])
    g2 = generate_mls(10, [1, 2, 5, 7, 8, 9], [ps[0] - 1, ps[1] - 1]) 
    return (g1 + g2) % 2
