from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RECORDINGS = 'Power'
GRID_FREQUENCY = {'A': 60, 'B': 50, 'C': 60, 'D': 50, 'E': 50, 'F': 50, 'G': 50, 'H': 50, 'I': 60}
GRID_50 = ['B', 'D', 'E', 'F', 'G', 'H']
GRID_60 = ['A', 'C', 'I']
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
    # TODO Delete
    # -----------------------------------------------------------------------------------
    TEST = 'NDDCDNNDAFANGBGBFCEHGHHHGHFDAIDNFHIIECBDENIBEFGNAGIINIGHAEFCCCFDGCECGIEICENBEEHADIHCGAABIHCNDBAGBFBB'
    POWER = [1, 3, 6, 7, 9, 10, 13, 14, 15, 17, 18, 19, 20, 22, 23, 29,
             30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47,
             49, 50, 52, 53, 56, 57, 58, 60, 64, 65, 66, 67, 68, 70, 71,
             72, 73, 76, 78, 85, 87, 88, 89, 90, 93, 95, 96, 97, 98]
    AUDIO = [2, 4, 5, 8, 11, 12, 16, 21, 24, 25, 26, 27, 28, 35,
             37, 43, 48, 51, 54, 55, 59, 61, 62, 63, 69, 74, 75,
             77, 79, 80, 81, 82, 83, 84, 86, 91, 92, 94, 99, 100]
    T = np.load('spectro_data/testing/focus_spectro_dataset.npy')
    # ---------------------------------------------------------------------------------

    #for TRAINING_GRID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
    for TRAINING_GRID in ['E']:
        # Load the training samples
        X = np.load(f'spectro_data/audio_aug/{RECORDINGS}/concatenated_data.npy')
        y = np.load(f'spectro_data/audio_aug/{RECORDINGS}/concatenated_labels.npy')

        # Load the models
        models = []
        models.append(joblib.load(f'Models/Final/GaussianNB/{RECORDINGS}/train_' + TRAINING_GRID + '_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/LogisticRegression/{RECORDINGS}/train_' + TRAINING_GRID + '_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/MLP/{RECORDINGS}/train_' + TRAINING_GRID + '_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/RandomForest/{RECORDINGS}/train_' + TRAINING_GRID + '_vs_ALL.pkl'))

        # Load the scaler
        scaler = joblib.load(f'Models/Final/Normalizer/{RECORDINGS}/scaler_' + TRAINING_GRID + '_vs_ALL.pkl')

        # TODO Delete
        # --------------------------------------------------------------------------
        Bindx = []
        for i in range(100):
            if TEST[i] == TRAINING_GRID and (i + 1) in POWER:  # ---------------------------------------------------------------------------------
                Bindx.append(i)
        # -------------------------------------------------------------------------

        # Create binary training set
        X, y = create_binary_set(X, y)

        # Normalize dataset
        # Flatten the images for easier processing
        X = X.reshape(X.shape[0], -1)
        X = scaler.transform(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

        y_pred_G = models[0].predict(X_val)
        y_pred_L = models[1].predict(X_val)
        y_pred_R = models[2].predict(X_val)
        y_pred_M = models[3].predict(X_val)

        y_pred = []
        for i in range(len(y_pred_G)):
            if y_pred_M[i] + y_pred_R[i] + y_pred_L[i] + y_pred_G[i] > 2:
                y_pred.append(1)
            elif y_pred_M[i] + y_pred_R[i] + y_pred_L[i] + y_pred_G[i] == 2:
                y_pred.append(y_pred_M[i])
            else:
                y_pred.append(0)

        accuracy = accuracy_score(y_val, y_pred)
        print('Model ', TRAINING_GRID)
        print('Val Accuracy on 80% training samples: ', accuracy)

        # TODO Delete
        # --------------------------------------------------------------------------------
        val_test = []
        for si in Bindx:
            for s in T[si]:
                s = s.reshape(1, -1)
                s = scaler.transform(s)

                y_pred_G = models[0].predict(s)
                y_pred_L = models[1].predict(s)
                y_pred_R = models[2].predict(s)
                y_pred_M = models[3].predict(s)
                if y_pred_M + y_pred_R + y_pred_L + y_pred_G > 2:
                    val_test.append(1)
                elif y_pred_M + y_pred_R + y_pred_L + y_pred_G == 2:
                    val_test.append(y_pred_M)
                else:
                    val_test.append(0)

        test80 = (val_test.count(1) / len(val_test))
        print('Test Accuracy on 80% training samples:  ', (val_test.count(1) / len(val_test)))
        # --------------------------------------------------------------------------------





        # Load the models
        models = []
        models.append(joblib.load(f'Models/Final/GaussianNB/{RECORDINGS}/{TRAINING_GRID}_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/LogisticRegression/{RECORDINGS}/{TRAINING_GRID}_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/MLP/{RECORDINGS}/{TRAINING_GRID}_vs_ALL.pkl'))
        models.append(joblib.load(f'Models/Final/RandomForest/{RECORDINGS}/{TRAINING_GRID}_vs_ALL.pkl'))

        # Load the scaler
        scaler = joblib.load(f'Models/Final/Normalizer/{RECORDINGS}/scaler_' + TRAINING_GRID + '_vs_ALL.pkl')


        y_pred_G = models[0].predict(X_val)
        y_pred_L = models[1].predict(X_val)
        y_pred_R = models[2].predict(X_val)
        y_pred_M = models[3].predict(X_val)

        y_pred = []
        for i in range(len(y_pred_G)):
            if y_pred_M[i] + y_pred_R[i] + y_pred_L[i] + y_pred_G[i] > 2:
                y_pred.append(1)
            elif y_pred_M[i] + y_pred_R[i] + y_pred_L[i] + y_pred_G[i] == 2:
                y_pred.append(y_pred_M[i])
            else:
                y_pred.append(0)

        accuracy = accuracy_score(y_val, y_pred)
        print('Val Accuracy on 80% training samples: ', accuracy)

        # TODO Delete
        # --------------------------------------------------------------------------------
        val_test = []
        for si in Bindx:
            for s in T[si]:
                s = s.reshape(1, -1)
                s = scaler.transform(s)

                y_pred_G = models[0].predict(s)
                y_pred_L = models[1].predict(s)
                y_pred_R = models[2].predict(s)
                y_pred_M = models[3].predict(s)
                if y_pred_M + y_pred_R + y_pred_L + y_pred_G >= 2:
                    val_test.append(1)
                #elif y_pred_M + y_pred_R + y_pred_L + y_pred_G == 2:
                #    val_test.append(y_pred_M)
                else:
                    val_test.append(0)

        test80 = (val_test.count(1) / len(val_test))
        print('Test Accuracy on 80% training samples:  ', (val_test.count(1) / len(val_test)))


