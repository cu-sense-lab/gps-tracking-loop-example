
from numpy import zeros, arange, exp, conj, pi, unravel_index, sqrt, log10
from numpy.fft import fft, ifft
from .utils import PSKSignal, sample_sequence


def coarse_acquire(samples, fs, f_center, signal: PSKSignal, doppler_bins, N, M=1, return_correlation=False):
    ''' Performs coarse acquisition algorithm
    
    Parameters
    ----------------------------------------------------------------------------
    samples: ndarray
        the sample buffer; acquisition will be at the start of the buffer
    fs: float
        the buffer sampling rate (Hz)
    f_center: float
        the center frequency of the front end (Hz)
    signal: PSKSignal
        signal to acquire
    doppler_bins: ndarray
        the doppler bins over which to search
    N: int
        the number of samples to coherently integrate
    M: int
        the number of blocks to non-coherently integrate

    
    Returns
    ----------------------------------------------------------------------------
    Output is a dictionary with following fields (valid at the start of the
    `samples` buffer):
        
        `code_phase` -- acquired code phase (in chips)
        `doppler_freq` -- acquired Doppler frequency shift
        `snr` -- acquired signal-to-noise ratio (dB)
    '''
    # unpack signal parameters
    code_seq = signal.code_seq
    f_carrier = signal.f_carrier
    f_code = signal.f_code
    
    # 2D array of correlation results
    #  dimensions are : (Doppler bins, code samples)
    correlation = zeros((len(doppler_bins), N))
    
    # time array for reference generation
    t = arange(N) / fs
        
    # phase-shift key symbols for our reference signal
    code_samples = exp(1j * pi * sample_sequence(t, code_seq, f_code))
    
    # intermediate frequency
    f_inter = f_carrier - f_center
    
    # FFT of blocks of `samples`, which we correlate with the reference signal
    fft_blocks = fft(samples[:N * M].reshape((M, N)), axis=1)
    
    # for each Doppler frequency, generate a reference and correlate with `samples`
    for i, doppler_freq in enumerate(doppler_bins):
        reference = code_samples * exp(1j * 2 * pi * (f_inter + doppler_freq) * t)
        correlation[i, :] = abs(ifft(conj(fft(reference)) * fft_blocks)).sum(axis=0) / (M * N)
    
    # At this point, the results of the correlation with the reference signal
    #  are stored in `correlation`.  Below, we perform a search for the peak
    #  of the correlation array
    
    # Calculate number of samples in one code period `N1`, then truncate 
    #  `correlation` accordingly
    code_length = len(code_seq)
    N1 = int(code_length * fs / f_code)
    correlation = correlation[:, :N1]
    
    # Find the row and column index of the maximum value of the correlation
    #  matrix;  these indices correspond to the Doppler bin `dopp_bin` and
    #  code sample `n0`, respectively
    dopp_bin, n0 = unravel_index(correlation.argmax(), correlation.shape)
    max_val = correlation[dopp_bin, n0]
    
    # Extract the acquired Doppler frequency and code phase
    doppler_freq = doppler_bins[dopp_bin]
    chip = (1 - n0 / N1) * code_length
    
    # Calculate the signal-to-noise ratio as the ratio of the correlation peak value
    #  to the noise standard deviation
    noise_std = sqrt(((correlation**2).sum() - max_val**2) / (M * N - 1))
    snr = 10 * log10(max_val / noise_std)
    
    outputs = {
        'n0': n0,
        'code_phase': chip,
        'doppler_freq': doppler_freq,
        'snr': snr,
        'correlation': correlation
    }
    return outputs

