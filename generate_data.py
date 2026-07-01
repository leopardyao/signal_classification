import numpy as np
from scipy.signal import butter, lfilter, savgol_filter
from scipy import signal

## define some tools signal generation
def add_noise(signal, noise_level=0.1):
    noise = noise_level * np.random.normal(size=len(signal))
    return signal + noise

def rotate_signal(signal, angle):
    radians = np.radians(angle)
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                    [np.sin(radians), np.cos(radians)]])
    rotated = np.matmul(np.vstack((np.arange(len(signal)),signal)).T, rotation_matrix)
    return rotated[:, 1]

def reverse_signal(signal):
    return signal[::-1]

def flip_signal(signal):
    return -signal

def normalize1d(data):
    return (data - min(data) ) / (max(data) - min(data))

def sub_min(data):
    return data - np.min(data)

def resample(data, size=100):
    return signal.resample(data, size)

## define functions to generate data
def make_group4(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:start+up+mid] = peak
    signal[start+up+mid:start+up+mid+down] = np.linspace(peak, 0, down, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group4_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(5, 31)
        up = np.random.randint(5, 26)
        mid = np.random.randint(5, 36)
        down = np.random.randint(5, 26)
        end = np.random.randint(5, 31)
        peak = np.random.randint(200, 600)
        sigma = min(max(-80, np.random.normal(0, 60)), 80)
        noise_level = np.random.randint(2, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = np.random.choice([0, 1], p=[0.8, 0.2])

        # print(peak, sigma)
        signal = make_group4(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 


def make_group6(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:start+up+mid] = peak
    signal[start+up+mid:start+up+mid+down] = np.linspace(peak, 0, down, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group6_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(5, 36)
        up = np.random.randint(3, 16)
        mid = np.random.randint(1, 3)
        down = np.random.randint(3, 16)
        end = np.random.randint(5, 36)
        peak = np.random.randint(200, 600)
        sigma = min(max(-80, np.random.normal(0, 60)), 80)
        noise_level = np.random.randint(2, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = np.random.choice([0, 1], p=[0.8, 0.2])

        signal = make_group6(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 


def make_group2(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:start+up+mid] = peak
    signal[start+up+mid:start+up+mid+down] = np.linspace(peak, 0, down, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group2_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(5, 21)
        up = np.random.randint(3, 11)
        mid = np.random.randint(1, 3)
        down = np.random.randint(25, 46)
        end = np.random.randint(3, 11)
        peak = np.random.randint(200, 600)
        sigma = min(max(-80, np.random.normal(0, 60)), 80)
        noise_level = np.random.randint(2, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = np.random.choice([0, 1], p=[0.8, 0.2])

        signal = make_group2(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 

def make_group3(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:start+up+mid] = peak
    signal[start+up+mid:start+up+mid+down] = np.linspace(peak, 0, down, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group3_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(25, 46)
        up = np.random.randint(5, 16)
        mid = np.random.randint(25, 46)
        down = 0
        end = 0
        peak = np.random.randint(200, 600)
        sigma = min(max(-80, np.random.normal(0, 60)), 80)
        noise_level = np.random.randint(2, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = np.random.choice([0, 1], p=[0.8, 0.2])

        signal = make_group3(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 


def make_group5(start, up, mid, down, end, up2, mid2, down2, end2, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end + up2 + mid2 + down2 + end2
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:start+up+mid] = peak
    signal[start+up+mid:start+up+mid+down] = np.linspace(peak, 0, down, endpoint=False)

    signal[start+up+mid+down+end:start+up+mid+down+end+up2] = np.linspace(0, peak, up2, endpoint=False)
    signal[start+up+mid+down+end+up2:start+up+mid+down+end+up2+mid2] = peak
    signal[start+up+mid+down+end+up2+mid2:start+up+mid+down+end+up2+mid2+down2] = np.linspace(peak, 0, down2, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group5_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(5, 31)
        up = np.random.randint(3, 16)
        mid = np.random.randint(1, 3)
        down = np.random.randint(3, 16)
        end = np.random.randint(5, 31)
        up2 = np.random.randint(3, 16)
        mid2 = np.random.randint(1, 3)
        down2 = np.random.randint(3, 16)
        end2 = np.random.randint(5, 31)
        peak = np.random.randint(200, 600)
        sigma = min(max(-80, np.random.normal(0, 60)), 80)
        noise_level = np.random.randint(2, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = np.random.choice([0, 1], p=[0.8, 0.2])

        signal = make_group5(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            up2=up2,
            mid2=mid2,
            down2=down2,
            end2=end2,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 

def make_group7(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group7_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(10, 46)
        up = np.random.randint(5, 46)
        mid = 0
        down = 0
        end = 0
        peak = np.random.randint(200, 401)
        sigma = min(max(-45, np.random.normal(0, 45)), 45)
        noise_level = np.random.randint(5, 10)
        reverse = np.random.choice([0, 1], p=[0.8, 0.2])
        flip = 0

        signal = make_group7(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 


def make_group1(start, up, mid, down, end, peak=200, sigma=0, noise_level=1, reverse=1, flip=1):
    length = start + up + mid + down + end
    signal = np.zeros(length)
    signal[start:start+up] = np.linspace(0, peak, up, endpoint=False)
    signal[start+up:] = peak

    signal = rotate_signal(signal, angle=sigma)

    signal = add_noise(signal, noise_level=noise_level)

    if reverse:
        signal = reverse_signal(signal)

    if flip:
        signal = flip_signal(signal)

    return signal

def generate_group1_signals(size=1000):
    signals = []
    for i in range(size):
        start = np.random.randint(0,3)
        up = np.random.randint(60, 91)
        mid = 0
        down = 0
        end = np.random.randint(0,3)
        peak = np.random.randint(20, 101)
        sigma = min(max(-45, np.random.normal(0, 45)), 45)
        noise_level = np.random.randint(3, 5)
        reverse = np.random.choice([0, 1], p=[0.5, 0.5])
        flip = 0

        signal = make_group1(
            start=start,
            up=up,
            mid=mid,
            down=down,
            end=end,
            peak=peak,
            sigma=sigma,
            noise_level=noise_level,
            reverse=reverse,
            flip=flip
        )

        signals.append(signal)

    return signals 