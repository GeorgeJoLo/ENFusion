import numpy as np
import joblib
from sklearn.metrics import accuracy_score

PRACTICE = 'AHCFFBGINDAFBDCINNAEHBBADCGNGBDDCHGEAIHIEHECFFNGEI'
TEST = 'NDDCDNNDAFANGBGBFCEHGHHHGHFDAIDNFHIIECBDENIBEFGNAGIINIGHAEFCCCFDGCECGIEICENBEEHADIHCGAABIHCNDBAGBFBB'

# MATLAB RESULTS
AUDIO = [2, 4, 5, 8, 11, 12, 16, 21, 24, 25, 26, 27, 28,  35,
         37, 43, 48, 51, 54, 55, 59, 61, 62, 63, 69, 74, 75,
         77, 79, 80, 81, 82, 83, 84, 86, 91, 92, 94, 99, 100]
POWER = [1, 3, 6, 7, 9, 10, 13, 14, 15, 17, 18, 19, 20, 22, 23, 29,
         30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47,
         49, 50, 52, 53, 56, 57, 58, 60, 64, 65, 66, 67, 68, 70, 71,
         72, 73, 76, 78, 85, 87, 88, 89, 90, 93, 95, 96, 97, 98]
FREQ_50 = [10, 100, 13, 14, 15, 16, 17, 19, 2, 20, 21, 22, 23, 24,
           25, 26, 28, 3, 31, 33, 34, 37, 39, 40, 41, 44, 45, 46,
           47, 5, 50, 55, 56, 58, 59, 6, 63, 64, 65, 67, 69, 7, 71,
           74, 76, 77, 78, 79, 8, 81, 83, 85, 88, 90, 93, 94, 96, 97, 98, 99]
FREQ_60 = [1, 11, 12, 18, 27, 29, 30, 32, 35, 36, 38, 4, 42, 43, 48,
           49, 51, 52, 53, 54, 57, 60, 61, 62, 66, 68, 70, 72, 73, 75,
           80, 82, 84, 86, 87, 89, 9, 91, 92, 95]

TESTING_FILE = 'spectro_data/testing/focus_spectro_dataset.npy'
GROUND_TRUTH = TEST  # PRACTICE
RECORDINGS = 'Power'  # 'Audio'
MODEL_DIRECTORY = 'Models/Final/MLP/' + RECORDINGS + '/'
SAMPLE_INDEXES = POWER if RECORDINGS == 'Power' else AUDIO
NORMALIZER_DIRECTORY = f'Models/Final/Normalizer/{RECORDINGS}/'


def predict_for_all_models(sample, sample_index, models, scalers):
    # Choose ENF models
    if sample_index in FREQ_60:
        models_indices = [0, 2, 8]  # A, C, I
    else:
        models_indices = [1, 3, 4, 5, 6, 7]  # B, D, E, F, G, H

    sample_predictions = []
    # For each model and scaler
    for ms_i in models_indices:
        model = models[ms_i]
        scaler = scalers[ms_i]

        model_predictions = []
        # For each part
        for p_i in range(len(sample)):
            part = sample[p_i]

            # Normalize
            x = part.reshape(1, -1)
            x = scaler.transform(x)

            # Predict
            model_predictions.append(model.predict_proba(x)[0])
        sample_predictions.append(model_predictions)

    return sample_predictions


def _aux_predict(model_idx, nof_models):
    if nof_models == 3:
        cases = {
            0: 'A',
            1: 'C',
            2: 'I'
        }
    else:
        cases = {
            0: 'B',
            1: 'D',
            2: 'E',
            3: 'F',
            4: 'G',
            5: 'H'
        }

    return cases.get(model_idx)


def predict_sample1(sample_predictions):
    # For each model
    # Set 1 if all part predictions are 1
    # Set 0 if all part predictions are 0
    # Set -1 if part predicitons are different
    model_predictions = []
    for mp in sample_predictions:
        if all(x == mp[0] for x in mp):
            model_predictions.append(mp[0])
        else:
            model_predictions.append(-1)

    # If exactly one model predicts 1
    # And max one model predicts -1
    # Final prediction in the grid of model with prediction 1
    if model_predictions.count(1) == 1 and model_predictions.count(-1) <= 1:
        return _aux_predict(model_predictions.index(1), len(sample_predictions))
    else:
        return 'N'


def predict_sample2(sample_predictions):
    # For each part
    # If more than one model predicts 1, it's Uknown
    # If no model predicts 1, it's Uknown
    # Else, it's the grid of the model
    part_predictions = []
    for pi in range(len(sample_predictions[0])):
        pps = [mp[pi] for mp in sample_predictions]

        if pps.count(1) == 1 and pps.count(0) == len(pps)-1:
            part_predictions.append(_aux_predict(pps.index(1), len(sample_predictions)))
        else:
            part_predictions.append('N')

    # If all parts are the same
    # And max one part is 'N'
    # Prediction is the grid of the rest parts (the non 'N')
    # Else, is 'N'
    if part_predictions.count('N') <= 1 and \
            ((part_predictions[0] != 'N' and all(x == part_predictions[0] or x == 'N' for x in part_predictions))
             or
             (part_predictions[1] != 'N' and all(x == part_predictions[1] or x == 'N' for x in part_predictions))):
        return part_predictions[0] if part_predictions[0] != 'N' else part_predictions[1]
    else:
        return 'N'


if __name__ == '__main__':
    # Load testing spectrograms
    data = np.load(TESTING_FILE)

    # =======================
    # Load Models
    # =======================
    # Load classifiers
    models = []
    for clf in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        models.append(joblib.load(MODEL_DIRECTORY + clf + '_vs_ALL.pkl'))

    # Load normalizers
    scalers = []
    for clf in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
        scalers.append(joblib.load(NORMALIZER_DIRECTORY + 'scaler_' + clf + '_vs_ALL.pkl'))

    # =======================
    # Predict for each sample
    # =======================
    temp = [None] * 100


    predictions = []
    for sample_index in SAMPLE_INDEXES:
        sample_predictions = predict_for_all_models(data[sample_index-1], sample_index, models, scalers)
        pred = predict_sample2(sample_predictions)
        predictions.append(pred)

        temp[sample_index-1] = sample_predictions
    np.save('MLP_pred_P.npy', temp, allow_pickle=True)

    # =======================
    # Evaluate
    # =======================
    # Find ground truth
    gt = []
    for i in SAMPLE_INDEXES:
        gt.append(GROUND_TRUTH[i-1])
    accuracy = accuracy_score(gt, predictions)
    print(accuracy)
