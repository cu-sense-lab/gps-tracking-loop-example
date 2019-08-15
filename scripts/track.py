#!/usr/bin/env python
# coding: utf-8

import os
import numpy
from numpy import zeros, arange, nan, real, imag, pi, sin, arctan, arctan2, angle, exp, sqrt, diff, pad, std
from scipy.io import loadmat, savemat
from gps_l1ca_utils import generate_GPS_L1CA_code
from acquisition_utils import coarse_acquire
from utils import PSKSignal, sample_sequence
from tracking_utils import compute_correlations


# Choose IF data file and appropriate data parameters
filepath = '../data/sim-RF_GPS-L1CA_5000_1250_complex_06s.mat'
data_type = 'simulated'
fs = 5e6                        # sampling rate (Hz)
f_center = 1.57542e9 - 1.25e6   # radio front-end center frequency (Hz)
prns = [4, 7, 10, 15, 29, 32]
prn = 10

# filepath = '../data/20190129_225529_gpsl1_tcxo_5000_0_4bit.mat'
# data_type = 'real-tcxo'
# fs = 5e6                        # sampling rate (Hz)
# f_center = 1.57542e9            # radio front-end center frequency (Hz)
# prns = [4, 5, 7, 8, 9, 16, 23, 27, 28, 30]
# prn = 4

# Load IF samples
IF_data = loadmat(filepath)
samples = IF_data['samples'][0]
T_data = len(samples) / fs      # duration of signal (s)

# Set parameters for GPS L1CA signal
f_carrier = 1.57542e9           # L1 carrier frequency (Hz)
f_code = 1.023e6                # L1 C/A code rate (chips/s)
f_inter = f_carrier - f_center  # intermediate frequency (Hz)


# Create signal and acquire
signal = PSKSignal(generate_GPS_L1CA_code(prn), 1.023e6, f_carrier)
doppler_bins = arange(-4000, 4000, 5)
coarse_acq_results = coarse_acquire(samples, fs, f_center, signal, doppler_bins, int(8e-3 * fs), 2, True)


# Set tracking loop parameters
T_blk = 2e-3                        # tracking block duration / integration period (s)
N_blk = int(T_blk * fs)             # number of samples per block
N_blocks = len(samples) // N_blk    # total number of blocks in the data
delay_spacing = 0.5                 # DLL correlator delay spacing
# Define tracking loop bandwidths
B_DLL = 2
B_PLL = 20


# Preallocate outputs
outputs = {
    key: nan * zeros(N_blocks) for key in [
        'code_phase',
        'measured_code_phase',
        'filtered_code_phase',
        'carrier_phase',
        'measured_carrier_phase',
        'filtered_carrier_phase',
        'doppler_freq',
        'measured_doppler_freq',
        'filtered_doppler_freq',
    ]
}
for key in ['early', 'prompt', 'late']:
    outputs[key] = nan * zeros(N_blocks, dtype=complex)
outputs['prn'] = prn
outputs['fs_IF'] = fs
outputs['f_center_IF'] = f_center
outputs['acq_correlation'] = coarse_acq_results['correlation']
outputs['acq_doppler_bins'] = doppler_bins
outputs['acq_snr'] = coarse_acq_results['snr']
outputs['n0'] = coarse_acq_results['n0']
outputs['code_phase0'] = coarse_acq_results['code_phase']
outputs['doppler_freq0'] = coarse_acq_results['doppler_freq']
outputs['time'] = arange(N_blocks) * T_blk
outputs['B_PLL'] = B_PLL
outputs['B_DLL'] = B_DLL
outputs['T'] = T_blk


# Set tracking state variables to coarse acquisition results
code_phase = coarse_acq_results['code_phase']
carrier_phase = 0
doppler_freq = coarse_acq_results['doppler_freq']

# Run tracking loop
for i in range(N_blocks): 
    # Get i-th block of samplescode_phase
    block = samples[i * N_blk:(i + 1) * N_blk]
    block -= numpy.mean(block)  # remove any DC bias
    
    # Obtain early, prompt, and late correlator outputs
    tracking_state = (code_phase, carrier_phase, doppler_freq)
    early, prompt, late = compute_correlations(block, fs, f_center, signal, tracking_state, [delay_spacing, 0, -delay_spacing])
    
    ### DLL ###
    # 1) Compute code phase error using early-minus-late discriminator
    code_phase_error = delay_spacing * (abs(early) - abs(late)) / (abs(early) + abs(late) + 2 * abs(prompt))
    
    # 2) Filter code phase error to reduce noise
    #  We implement the DLL filter by updating code phase in proportion to code
    #  phase dicriminator output.  The result has the equivalentresponse of a
    #  1st-order DLL filter
    filtered_code_phase_error = T_blk * B_DLL / .25 * code_phase_error
    
    measured_code_phase = code_phase + code_phase_error
    filtered_code_phase = code_phase + filtered_code_phase_error
    
    ### PLL ###
    # 1) Compute phase error (in cycles) using appropriate phase discriminator
    delta_theta = arctan(imag(prompt) / real(prompt)) / (2 * pi)
#     delta_theta = arctan2(imag(prompt), real(prompt)) / (2 * pi)
    carrier_phase_error = delta_theta
    doppler_freq_error = T_blk / 2 * delta_theta
    
    # 2) Filter carrier phase error to reduce noise
    #  We implement the PLL filter by updating carrier phase and frequency in
    #  proportion to the phase discriminator output in a way that has the
    #  equivalent response to a 2nd-order PLL filter
    zeta = 1 / sqrt(2)
    omega_n = B_PLL / .53
    filtered_carrier_phase_error = (2 * zeta * omega_n * T_blk - 3 / 2 * omega_n**2 * T_blk**2) * delta_theta
    filtered_doppler_freq_error = omega_n**2 * T_blk * delta_theta
    
    measured_carrier_phase = carrier_phase + carrier_phase_error
    filtered_carrier_phase = carrier_phase + filtered_carrier_phase_error
    
    measured_doppler_freq = doppler_freq + doppler_freq_error
    filtered_doppler_freq = doppler_freq + filtered_doppler_freq_error
    
    # Write outputs
    outputs['early'][i] = early
    outputs['prompt'][i] = prompt
    outputs['late'][i] = late
    outputs['code_phase'][i] = code_phase
    outputs['measured_code_phase'][i] = measured_code_phase
    outputs['filtered_code_phase'][i] = filtered_code_phase
    outputs['carrier_phase'][i] = carrier_phase
    outputs['measured_carrier_phase'][i] = measured_carrier_phase
    outputs['filtered_carrier_phase'][i] = filtered_carrier_phase
    outputs['doppler_freq'][i] = doppler_freq
    outputs['measured_doppler_freq'][i] = measured_doppler_freq
    outputs['filtered_doppler_freq'][i] = filtered_doppler_freq
    
    # Update to next time epoch (this step is considered part of the loop filter!)
    code_phase = filtered_code_phase
    carrier_phase = filtered_carrier_phase
    doppler_freq = filtered_doppler_freq
    
    #  Here we apply carrier-aiding by adjusting `f_code` based on Doppler frequency
    f_code_adj = signal.f_code * (1 + doppler_freq / signal.f_carrier)
    code_phase += f_code_adj * T_blk
    f_inter = signal.f_carrier - f_center
    carrier_phase += (f_inter + doppler_freq) * T_blk


# Remove nominal code rate and intermediate frequency from outputs
#  This makes it easier to:
#    1) compare code phase to a real receiver's pseudorange outputs
#    2) compare carrier phase output between datasets with different intermediate frequencies
t = arange(N_blocks) * T_blk
outputs['code_phase'] -= t * f_code
outputs['measured_code_phase'] -= t * f_code
outputs['filtered_code_phase'] -= t * f_code
outputs['carrier_phase'] -= t * f_inter
outputs['measured_carrier_phase'] -= t * f_inter
outputs['filtered_carrier_phase'] -= t * f_inter


# If data was simulated, we should also store the true state values
if data_type == 'simulated':
    prns = IF_data['prns'][0]
    chips = IF_data['chips']
    chips = dict(zip(prns, chips))
    code_phase_truth = (chips[prn][::N_blk])[:N_blocks]
    carrier_phase_truth = code_phase_truth * f_carrier / signal.f_code
    carrier_phase_truth -= carrier_phase_truth[0] - carrier_phase_truth[0] % 1
    doppler_freq_truth = pad(diff(carrier_phase_truth), (0, 1), 'edge') / T_blk

    outputs['code_phase_truth'] = code_phase_truth
    outputs['carrier_phase_truth'] = carrier_phase_truth
    outputs['doppler_freq_truth'] = doppler_freq_truth


output_dir = '../tracking-output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filepath = os.path.join(output_dir, '{0}_PRN-{1:02}_PLL-BW-{2:02}.mat'.format(data_type, prn, B_PLL))
savemat(output_filepath, outputs)




