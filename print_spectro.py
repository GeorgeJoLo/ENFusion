import re
import nlpaug.augmenter.spectrogram as nas
import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import nlpaug.augmenter.audio as naa


GRID_ENF = {'A': 60, 'B': 50, 'C': 60, 'D': 50, 'E': 50, 'F': 50, 'G': 50, 'H': 50, 'I': 60}

FS = 1000

N_FM = 2
N_TM = 2

PART_LEN = 5 * 60
OVERLAP_LEN = 2 * 60

HOP_LEN = 2 ** 8
NFFT = 2 ** 16
WINDOW = 'hann'

MARGIN = 0.5
FACTOR = (20, 40)


def iterate_dataset(dataset_directory):
    """
    Iterates to each wav file in the dataset and applies a function
    """
    spectro_data = []
    spectro_label = []
    for grid_folder in os.listdir(dataset_directory):
        grid_folder_path = os.path.join(dataset_directory, grid_folder)
        if grid_folder != 'Grid_D':
            continue
        if not os.path.isdir(grid_folder_path):
            continue

        for data_type in ['Power_recordings']:  # ['Audio_recordings', 'Power_recordings']
            data_folder_path = os.path.join(grid_folder_path, data_type)
            if not os.path.exists(data_folder_path):
                continue

            for root, _, files in os.walk(data_folder_path):
                for file in files:
                    if file.endswith('.wav'):
                        print(file)
                        file_path = os.path.join(root, file)
                        data, labels = create_augmented_spectro_data(file_path, N_FM, N_TM)
                        spectro_data.extend(data)
                        spectro_label.extend(labels)
    np.savez('spectro_data/audio_aug/focus_spectro_dataset_I_A_white.npz', data=np.array(spectro_data), labels=np.array(spectro_label))


def extract_sample_label(file_path):
    # Define a regular expression pattern to match "X" in the string
    pattern = r"Train_Grid_([\w\d]+)_"

    # Search for the pattern in the input string
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the value of "X"
    if match:
        return match.group(1)
    else:
        # If no match is found, you can handle it accordingly, e.g., return None
        return None


def separate_wav_file(wav_file_path, duration, overlap):
    """
    Separates a wav file into smaller parts and returns the parts

    Parameters
    ---------------
    duration: int
        The duration of each part (in seconds)
    overlap:  int
        The duration of the overlap between the fragments (in seconds)
    wav_file_path: str
    """
    # Create the array to store th parts
    parts = []

    # Read the WAV file
    data, fs = librosa.load(wav_file_path, sr=None)
    # fs, data = wavfile.read(wav_file_path)

    # Convert M and F from seconds to nof_samples
    duration_samples = duration * fs
    shift_samples = (duration - overlap) * fs

    start_sample = 0
    end_sample = duration_samples
    i = 0

    while end_sample < len(data):
        # Extract the part from the audio
        part = data[start_sample:end_sample]

        # Store the part
        parts.append(part)

        start_sample += shift_samples
        end_sample += shift_samples
        i += 1

    # Create the last part containing the last minutes of the audio
    if start_sample < len(data):
        end_sample = len(data)
        start_sample = end_sample - duration_samples

        # Extract the part from the audio
        part = data[start_sample-1:end_sample - 1]

        # Store the part
        parts.append(part)

    return np.array(parts)


def spectrogram(y, hop_length, n_fft, window, center=True):
    D_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    D_amplitude = np.abs(D_complex)
    return D_amplitude


def focus_ENF(spectrogram, focus_freq, margin, fs, n_fft):
    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=fs, n_fft=n_fft)

    # Find the index of the frequency bin closest to the focus frequency (e.g., 50 Hz)
    focus_idx = np.argmin(np.abs(freqs - focus_freq))

    # Extract frequency bins around the focus frequency within the margin (e.g., 1 Hz around 50 Hz)
    start_idx = np.argmin(np.abs(freqs - (focus_freq - margin)))
    end_idx = np.argmin(np.abs(freqs - (focus_freq + margin)))

    D_focus = spectrogram[start_idx+1:end_idx-1]

    return D_focus


def create_augmented_spectro_data(file_path, n_fm, n_tm):
    # Find the label of the sample
    label = extract_sample_label(file_path)

    # Separate the wav file
    parts = separate_wav_file(file_path, PART_LEN, OVERLAP_LEN)

    spectro_data = []
    spectro_labels = [label] * (len(parts) * (1+n_tm+n_fm))
    # For each part
    for part in parts:

        # Calculate the spectrogram
        spectro = spectrogram(part, HOP_LEN, 2**18, WINDOW)
        spectro = focus_ENF(spectro, GRID_ENF[label], 1, FS, 2**18)
        spectro_data.append(spectro)

        D_db_focus_original = librosa.amplitude_to_db(spectro, ref=np.max)
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(D_db_focus_original, sr=1000, hop_length=2 ** 8, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original Spectrogram for element {1}')
        plt.show()

    return spectro_data, spectro_labels


if __name__ == '__main__':
    iterate_dataset('Dataset/')
