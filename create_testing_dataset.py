import re
import nlpaug.augmenter.spectrogram as nas
import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
import nlpaug.augmenter.audio as naa


FREQ_50 = [10, 100, 13, 14, 15, 16, 17, 19, 2, 20, 21, 22, 23, 24,
           25, 26, 28, 3, 31, 33, 34, 37, 39, 40, 41, 44, 45, 46,
           47, 5, 50, 55, 56, 58, 59, 6, 63, 64, 65, 67, 69, 7, 71,
           74, 76, 77, 78, 79, 8, 81, 83, 85, 88, 90, 93, 94, 96, 97, 98, 99]
FREQ_60 = [1, 11, 12, 18, 27, 29, 30, 32, 35, 36, 38, 4, 42, 43, 48,
           49, 51, 52, 53, 54, 57, 60, 61, 62, 66, 68, 70, 72, 73, 75,
           80, 82, 84, 86, 87, 89, 9, 91, 92, 95]

DIRECTORY = 'SPS_CUP_2016_Data/Testing_dataset/'
NOF_SAMPLES = 100

FS = 1000

PART_LEN = 5 * 60
OVERLAP_LEN = 0

HOP_LEN = 2 ** 8
NFFT = 2 ** 16
WINDOW = 'hann'

MARGIN = 0.5


def iterate_dataset(dataset_directory):
    """
    Iterates to each wav file in the dataset and applies a function
    """
    spectro_data = [None] * NOF_SAMPLES
    for root, _, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith('.wav'):
                print(file)
                file_path = os.path.join(root, file)
                sample_index = extract_sample_index(file_path)
                data = create_spectro_data(file_path, 50 if sample_index in FREQ_50 else 60)
                spectro_data[sample_index-1] = data
   # np.savez('spectro_data/testing/focus_spectro_dataset.npy', np.array(spectro_data))


def extract_sample_index(file_path):
    # Define a regular expression pattern to match "X" in the string
    pattern = r"Test_([\w\d]+).wav"  # Change to r"Practice_([\w\d]+).wav"

    # Search for the pattern in the input string
    match = re.search(pattern, file_path)

    # If a match is found, extract and return the value of "X"
    if match:
        return int(match.group(1))
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


def create_spectro_data(file_path, focus_freq):
    # Separate the wav file
    parts = separate_wav_file(file_path, PART_LEN, OVERLAP_LEN)

    spectro_data = []
    # For each part
    for part in parts:
        # Calculate the spectrogram
        spectro = spectrogram(part, HOP_LEN, NFFT, WINDOW)
        spectro = focus_ENF(spectro, focus_freq, MARGIN, FS, NFFT)

        D_db_focus_original = librosa.amplitude_to_db(spectro, ref=np.max)
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(D_db_focus_original, sr=1000, hop_length=2 ** 8, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original Spectrogram for element {1}')
        plt.show()

        spectro_data.append(spectro)




    return spectro_data


if __name__ == '__main__':
    iterate_dataset(DIRECTORY)
