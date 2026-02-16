import numpy as np
from utils import load_wav_data
import random
from scipy.ndimage import gaussian_filter1d


# =========================================================
# Utility
# =========================================================

def _copy(spec):
    return np.array(spec, copy=True)


# =========================================================
# 1) Relative Gaussian Noise
# =========================================================

def add_relative_noise(spec, scale=0.03):
    """
    Adds Gaussian noise relative to spectrogram std.
    Good for simulating mic noise / breath.
    """
    spec = _copy(spec)
    noise = np.random.randn(*spec.shape)
    return spec + scale * np.std(spec) * noise


# =========================================================
# 2) Realistic Silence Generator
# =========================================================

def generate_realistic_silence(freq_bins, length):
    """
    Generates low-level smooth spectral noise instead of flat -80.
    """
    base = np.random.normal(loc=-80, scale=2.0, size=(freq_bins, length))
    smooth = gaussian_filter1d(base, sigma=3, axis=0)
    return smooth


# =========================================================
# 3) Time Mask (SpecAugment style)
# =========================================================

def time_mask(spec, max_width=20):
    spec = _copy(spec)
    t = spec.shape[0]

    width = np.random.randint(0, max_width + 1)
    if width == 0 or width >= t:
        return spec

    start = np.random.randint(0, t - width)
    spec[start:start+width, :] = np.min(spec)
    return spec


# =========================================================
# 4) Frequency Mask
# =========================================================

def freq_mask(spec, max_width=40):
    spec = _copy(spec)
    f = spec.shape[1]

    width = np.random.randint(0, max_width + 1)
    if width == 0 or width >= f:
        return spec

    start = np.random.randint(0, f - width)
    spec[:, start:start+width] = np.min(spec)
    return spec


# =========================================================
# 5) Smooth Spectral Envelope (VERY IMPORTANT)
# =========================================================

def random_spectral_envelope(spec, strength=0.5, smooth_sigma=40):
    """
    Applies smooth frequency coloration.
    Simulates mouth shape / table material / mic EQ.
    """
    spec = _copy(spec)

    freq_bins = spec.shape[1]

    curve = np.random.randn(freq_bins)
    curve = gaussian_filter1d(curve, sigma=smooth_sigma)

    curve /= (np.max(np.abs(curve)) + 1e-8)
    curve *= strength

    return spec + curve


# =========================================================
# 6) Random Gain Scaling
# =========================================================

def random_gain(spec, low=0.7, high=1.3):
    spec = _copy(spec)
    gain = np.random.uniform(low, high)
    return spec * gain


# =========================================================
# 7) Random Time Shift
# =========================================================

def random_time_shift(spec, max_shift=15):
    """
    Circular shift in time dimension.
    """
    spec = _copy(spec)
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(spec, shift, axis=0)


# =========================================================
# 8) Mild Dynamic Range Distortion
# =========================================================

def random_dynamic_range(spec, gamma_low=0.8, gamma_high=1.2):
    """
    Slight compression / expansion of dynamics.
    """
    spec = _copy(spec)
    gamma = np.random.uniform(gamma_low, gamma_high)
    return np.sign(spec) * (np.abs(spec) ** gamma)

# =========================================================
# 9) Add recorded background noise to training samples
# =========================================================

def add_background_noise(audio_data, path_to_noise):

    directory = 'background-sounds'
    background_noise, _ = load_wav_data(directory)
    background_noise = np.array(audio_array)

    len_windows = audio_data.shape[1]
    
    for i in range(audio_data.shape[0]):

        # Create subplots
        fig, axs = plt.subplots(2, figsize=(12, 8))

        # Plot the spectrogram on the first subplot
        img = axs[0].imshow(stft.T, aspect='auto', origin='lower', cmap='jet')

        noise_sample_ind = random.randint(0, background_noise.shape[1]-len_windows-1)
        noise_sample = background_noise[0, noise_sample_ind:noise_sample_ind+len_windows, :]
        audio_data[i,:,:] += noise_sample

        img = axs[1].imshow(stft.T, aspect='auto', origin='lower', cmap='jet')

        # Show the plot
        plt.show()

# =========================================================
# 10) Full Augmentation Pipeline
# =========================================================

def augment_spectrogram(
    spec,
    noise_prob=0.5,
    time_mask_prob=0.3,
    freq_mask_prob=0.3,
    envelope_prob=0.7,
    gain_prob=0.5,
    shift_prob=0.5,
    dyn_range_prob=0.3
):
    """
    Apply random augmentation sequence.
    Input shape: (time, freq)
    """

    spec_aug = _copy(spec)

    if np.random.rand() < noise_prob:
        spec_aug = add_relative_noise(spec_aug)

    if np.random.rand() < time_mask_prob:
        spec_aug = time_mask(spec_aug)

    if np.random.rand() < freq_mask_prob:
        spec_aug = freq_mask(spec_aug)

    if np.random.rand() < envelope_prob:
        spec_aug = random_spectral_envelope(spec_aug)

    if np.random.rand() < gain_prob:
        spec_aug = random_gain(spec_aug)

    if np.random.rand() < shift_prob:
        spec_aug = random_time_shift(spec_aug)

    if np.random.rand() < dyn_range_prob:
        spec_aug = random_dynamic_range(spec_aug)

    return spec_aug
