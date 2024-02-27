import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load your audio file
y, sr = librosa.load('/media/blue/ckorgial/ENF_Autoencoder/Dataset/Grid_G/Audio_recordings/Train_Grid_G_A1.wav',
                     sr=None)


def display_and_save_spectrogram(y, sr, hop_length, n_fft, window, center=True, filename='spectogram.npy',
                                 focus_freq=60, margin=1):
    D_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=center)
    D_amplitude = np.abs(D_complex)

    # Get frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Find the index of the frequency bin closest to the focus frequency (e.g., 50 Hz)
    focus_idx = np.argmin(np.abs(freqs - focus_freq))

    # Extract frequency bins around the focus frequency within the margin (e.g., 1 Hz around 50 Hz)

    # start_idx = np.argmin(np.abs(freqs - (focus_freq - margin)))
    # end_idx = np.argmin(np.abs(freqs - (focus_freq + margin)))

    D_focus = D_amplitude  # [focus_idx-2:focus_idx + 3]

    # Save the focused spectrogram to a .npy file
    '''np.save(filename, D_focus)'''

    #return D_focus

    # Display the focused spectrogram
    D_db_focus = librosa.amplitude_to_db(D_focus, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D_db_focus, sr=sr, hop_length=hop_length, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram around {focus_freq} Hz')
    plt.tight_layout()
    plt.show()


# Adjust these parameters as per your requirement and experiments
hop_length = 512  # Number of samples between successive frames
n_fft = 2048  # Length of the FFT window
window = 'hann'  # Type of window function

display_and_save_spectrogram(y, sr, hop_length, n_fft, window)
