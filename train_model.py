from collections import Counter

import joblib
import numpy as np
import keras
from onedal.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from keras.models import Sequential
# from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard


GRID_FREQUENCY = {'A': 60, 'B': 50, 'C': 60, 'D': 50, 'E': 50, 'F': 50, 'G': 50, 'H': 50, 'I': 60}
GRID_50 = ['B', 'D', 'E', 'F', 'G', 'H']
GRID_60 = ['A', 'C', 'I']

# Data Splitting
RECORDINGS = 'Power'
TRAINING_GRID = 'D'

ENF_GRID = GRID_50 if TRAINING_GRID in GRID_50 else GRID_60

TEST_SIZE = 0.2


def one_vs_all(X, y):
    class_counts = Counter(y)
    label_class_size = class_counts[TRAINING_GRID]

    # Create arrays to store the new balanced dataset
    X_balanced = []
    y_balanced = []

    # Keep all instances of label class
    label_class_indicies = np.where(y == TRAINING_GRID)[0]
    X_balanced.extend(X[label_class_indicies])
    y_balanced.extend(y[label_class_indicies])

    # Keep an equal amount of instances from each remaining class
    rest_class_size = label_class_size // (len(class_counts) - 1)
    for class_label, count in class_counts.items():
        if class_label != TRAINING_GRID:
            indices = np.where(y == class_label)[0]
            selected_indices = np.random.choice(indices, size=rest_class_size, replace=False)
            X_balanced.extend(X[selected_indices])
            y_balanced.extend(y[selected_indices])

    # Change labels to binary classification problem
    y_balanced = np.array(y_balanced)
    y_balanced = np.where(y_balanced == TRAINING_GRID, 1, 0)

    # Convert balanced lists to numpy arrays
    X_balanced = np.array(X_balanced)

    return X_balanced, y_balanced


def create_binary_set(X, y):
    # If nominal ENF frequency is 60
    if GRID_FREQUENCY[TRAINING_GRID] == 60:
        # Keep only samples with nominal ENF frequency 60
        same_freq_indc = np.where((y == 'A') | (y == 'C') | (y == 'I'))[0]
        X_ = X[same_freq_indc]
        y_ = y[same_freq_indc]

        X_grid, y_grid = one_vs_all(X_, y_)
    # If nominal ENF frequency is 50
    else:
        # Keep only samples with nominal ENF frequency 50
        same_freq_indc = np.where((y == 'B') | (y == 'D') | (y == 'E') | (y == 'F') | (y == 'G') | (y == 'H'))[0]
        X_ = X[same_freq_indc]
        y_ = y[same_freq_indc]

        X_grid, y_grid = one_vs_all(X_, y_)

    return X_grid, y_grid


if __name__ == '__main__':
    # Load the training samples
    X = np.load(f'spectro_data/audio_aug/{RECORDINGS}/concatenated_data.npy')
    y = np.load(f'spectro_data/audio_aug/{RECORDINGS}/concatenated_labels.npy')
    T = np.load('spectro_data/testing/focus_spectro_dataset.npy')

    TEST = 'NDDCDNNDAFANGBGBFCEHGHHHGHFDAIDNFHIIECBDENIBEFGNAGIINIGHAEFCCCFDGCECGIEICENBEEHADIHCGAABIHCNDBAGBFBB'
    POWER = [1, 3, 6, 7, 9, 10, 13, 14, 15, 17, 18, 19, 20, 22, 23, 29,
             30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47,
             49, 50, 52, 53, 56, 57, 58, 60, 64, 65, 66, 67, 68, 70, 71,
             72, 73, 76, 78, 85, 87, 88, 89, 90, 93, 95, 96, 97, 98]
    AUDIO = [2, 4, 5, 8, 11, 12, 16, 21, 24, 25, 26, 27, 28, 35,
             37, 43, 48, 51, 54, 55, 59, 61, 62, 63, 69, 74, 75,
             77, 79, 80, 81, 82, 83, 84, 86, 91, 92, 94, 99, 100]
    Bindx = []
    for i in range(100):
        if TEST[i] == TRAINING_GRID and (i+1) in POWER:
            Bindx.append(i)


    # Create binary training set
    X, y = create_binary_set(X, y)

    # Normalize dataset
    # Flatten the images for easier processing
    X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, f'scaler_{TRAINING_GRID}_vs_ALL.pkl')

    # Split into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # Choose a Machine Learning Model
    # In this example, we'll use Logistic Regression as the classifier. You can replace it with your preferred model.
    #model = LogisticRegression(max_iter=1000)
    model = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Step 3: Evaluate the Model
    # Make predictions on the test data
    y_pred = model.predict(X_val)

    # Evaluate the model using metrics
    accuracy = accuracy_score(y_val, y_pred)
    print('Random Forest for Model: ', TRAINING_GRID)
    print('Val Accuracy on 80% training samples:   ', accuracy)

    val_test = []
    for si in Bindx:
        for s in T[si]:
            s = s.reshape(1, -1)
            s = scaler.transform(s)

            p = model.predict(s)
            val_test.append(p)
    test80 = (val_test.count(1) / len(val_test))
    print('Test Accuracy on 80% training samples:  ', (val_test.count(1) / len(val_test)))

    shuffled_indices = np.random.permutation(len(y))
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    final_model = RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42)
    final_model.fit(X, y)
    y_pred = final_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print('Val Accuracy on 100% training samples:  ', accuracy)

    val_test = []
    for si in Bindx:
        for s in T[si]:
            s = s.reshape(1, -1)
            s = scaler.transform(s)

            p = final_model.predict(s)
            val_test.append(p)
    test100 = (val_test.count(1) / len(val_test))
    print('Test Accuracy on 100% training samples: ', (val_test.count(1) / len(val_test)))

    model_filename = f"Models/Final/RandomForest/{RECORDINGS}/{TRAINING_GRID}_vs_ALL.pkl"
    if test100 > test80:
        print("FInal")
        joblib.dump(final_model, model_filename)
    else:
        print("Val")
        joblib.dump(model, model_filename)
