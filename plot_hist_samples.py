import numpy as np
from matplotlib import pyplot as plt

lp = np.load("/media/blue/gtzolopo/ENF_Autoencoder2/spectro_data/audio_aug/Power/concatenated_labels.npy")
la = np.load("/media/blue/gtzolopo/ENF_Autoencoder2/spectro_data/audio_aug/Audio/concatenated_labels.npy")


def foo(arr):
    # Find unique values and their counts
    unique_values, counts = np.unique(arr, return_counts=True)

    # Create a dictionary with the counts
    letter_count_dict = dict(zip(unique_values, counts))

    # Print the dictionary
    return letter_count_dict

dict1 = foo(lp)
dict2 = foo(la)

# Extract keys and values
keys = list(dict1.keys())
values1 = list(dict1.values())
values2 = list(dict2.values())

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = np.arange(len(keys))
r2 = [x + bar_width for x in r1]

# Plotting the bars
plt.bar(r1, values1, color='#1f77b4', width=bar_width, edgecolor='grey', label='Power')
plt.bar(r2, values2, color='#ff7f0e', width=bar_width, edgecolor='grey', label='Audio')

# Adding labels
plt.xlabel('Grids')
plt.ylabel('Samples per Grid')
plt.xticks([r + bar_width/2 for r in range(len(keys))], keys)

# Adding legend
plt.legend(loc='center right')
plt.savefig('sample_hist.png')
# Show the plot
plt.show()



