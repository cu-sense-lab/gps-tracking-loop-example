from numpy import array, zeros, arange, roll, exp, conj, pi, sum
from .utils import sample_sequence, PSKSignal

def compute_correlations(block, fs, f_center, signal, state, code_phase_shifts=[0]):
    '''
    Performs matched-filter correlations between input data in `block` and GNSS
    signal defined in `signal` with signal parameters given in `state`.
    
    Parameters
    ----------------------------------------------------------------------------
    block: ndarray
        GNSS data samples (e.g. from RF front-end)
    fs: float
        sampling frequency (Hz) of the `block` sample array
    f_center: float
        center frequency of samples provided in `block`
    signal: PSKSignal object or equivalent; contains the following fields:
        f_carrier: float; signal's nominal carrier frequency (Hz)
        code_seq: array; contains code phase-shift keys in units of half-cycles
        f_code: float; the code rate (chips/s)
    state: sequence (tuple, list, etc) containing these elements (in order):
        code_phase: float; code phase estimate (chips)
        carrier_phase: float; carrier phase estimate (cycles)
        doppler_freq: float; Doppler frequency estimate (Hz)
    code_phase_shifts: list of floats
        code phase shifts for which to compute correlator outputs.  Defaults to
        `[0]` so that only the prompt correlation is computed.
    
    Returns a list of correlations between the data in `block` and the reference
        signal defined by `signal` and `state` for each code phase offset given
        in `code_phase_shifts`
    '''
    # Unpack the required signal and state variables
    f_carrier = signal.f_carrier
    code_seq = signal.code_seq
    f_code = signal.f_code
    code_phase = state[0]
    carrier_phase = state[1]
    doppler_freq = state[2]
    
    # Adjust code rate based on Doppler frequency
    #  i.e. carrier-aiding
    #  not actually necessary for reasonable integration periods
    f_code = f_code * (1 + doppler_freq / f_carrier)
    
    # Compute intermediate frequency
    f_inter = f_carrier - f_center
    
    t = arange(len(block)) / fs
    
    correlations = []
    for code_phase_shift in code_phase_shifts:
        
        # Generate reference
        #  BPSK code samples (note: `exp(j*pi*0) = 1` and `exp(j*pi*1) = -1`)
        code_samples = 1 - 2 * sample_sequence(t, code_seq, f_code, code_phase + code_phase_shift)
        reference_carrier = code_samples * exp(1j * 2 * pi * ((f_inter + doppler_freq) * t + carrier_phase))

        # Correlate
        #  wipe off carrier and code and integrate
        correlation = sum(block * conj(reference_carrier)) / len(block)
        
        correlations.append(correlation)
    
    return correlations

