
from numpy import ndarray, floor


def sample_sequence(t, seq, fc, c0=0):
    '''Generates samples of a sequence given sampling times and a sequence with
    a specified rate and initial phase (optional).
    
    Parameters
    ----------------------------------------------------------------------------
    t : ndarray of shape (N,)
        times at which to sample the sequence
    seq : ndarray of shape (N,)
        sequence to sample
    fc : float
        symbol rate of the sequence
    c0 : float
        (optional) defaults to zero -- the initial phase of the sampled sequence

    Returns
    ----------------------------------------------------------------------------
    output : ndarray of shape (N,)
        the sequence samples
    '''
    seq_indices = floor(c0 + t * fc) % len(seq)
    return seq[seq_indices.astype(int)]


class PSKSignal:
    '''
    This class describes objects that hold necessary information to define a
    simple phase-shift keying (PSK) signal.
    
    code_seq: float ndarray of shape (N,)
        the sequence of phase-shift symbols, in units of cycles
    f_code: float
        the modulation rate (Hz) of `code_seq` onto the carrier
    f_carrier: float
        the carrier frequency (Hz)
    '''
    def __init__(self, code_seq, f_code, f_carrier):
        self.code_seq = code_seq
        self.f_code = f_code
        self.f_carrier = f_carrier
