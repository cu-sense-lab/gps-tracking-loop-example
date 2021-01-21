from numpy import sqrt, zeros, floor, iscomplex, real, imag, max, abs, any
from numpy.random import randn

def add_noise(signals, cn0s=[49.], tk=535., rx_bandwidth=2e6):
    '''
    Adds noise to and combines a set of simulated RF signals.
    ----------
    signals: list of ndarrays of shape (N,)
        signal vectors that will be modified by an amplitude term and added together
    cn0s: list of floats
        signal-to-noise ratios (in dbHz), which typically ranges between 33-55 dbHz,
        for corresponding signal in signals list. (nominal 49)
    tk: float
        receiver system noise temperature, which is a combination of sky noise and
        thermal noise in the receiver. (default 535)
    rx_bandwidth: float
        receiver bandwidth (default 2MHz)
    -----
    '''
    # define Boltzmann's constant
    k = 1.38e-23
    noise_pwr = k * tk
    # calculate signal amplitude using relationship Ps = 1/2 A^2 and
    # CN0 = 10 * log(Ps/Pn) with Ps in Watts and Pn in Watts/Hz
    # `Ps` defined this way because `sum(cos(t)**2)/len(t)` equals 1/2
    amplitudes = [sqrt(2 * noise_pwr * 10 ** (cn0 / 10.)) for cn0 in cn0s]
    samples = zeros(signals[0].shape, dtype=signals[0].dtype)
    N = len(samples)
    for signal, a in zip(signals, amplitudes):
        samples += a * signal
    # noise amplitude depends on receiver bandwidth
    noise_var = noise_pwr * rx_bandwidth
    return samples + sqrt(noise_var) / 2 * (randn(N) + 1j * randn(N))


def quantize(signal, bits=4, ADC_range=None):
    '''
    Simulates quantization of a signal.
    
    Parameters
    ----------------------------------------------------------------------------
    signal: ndarray of shape (N,)
    bits: int
        bits in ADC--the number of signal quantization levels will be 2^`bits`
    ADC_range: 2-tuple of floats (min, max) or None
        the minimum and maximum values of the output quantized signal, or if
        `None` (default), then the effective ADC range is +/- `max(abs(signal))`
        
    
    Returns
    ----------------------------------------------------------------------------
    output: ndarray of
        the quantized signal
    '''
    levels = 2**bits - 1
    i = floor(levels * real(signal) / max(abs(signal)))
    if any(iscomplex(signal)):
        q = floor(levels * imag(signal) / max(abs(signal)))
        return i + 1j * q
    return i