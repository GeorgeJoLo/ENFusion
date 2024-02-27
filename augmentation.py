import nlpaug.augmenter.spectrogram as nas
import numpy as np

# Load your saved spectrogram (make sure it's saved as 'spectrogram.npy')
data = np.load('spectogram.npy')

# Create a FrequencyMaskingAugmenter
aug = nas.FrequencyMaskingAug()

# Augment the spectrogram
augmented_data = aug.augment(data, 2)

# You can save the augmented data to a new .npy file or use it as needed
np.save('augmented_spectogram22.npy', augmented_data)
