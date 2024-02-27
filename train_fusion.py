import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

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

RECORDINGS = 'Power'  # 'Audio'
ENF = 60

RECORDING_INDEXES = POWER if RECORDINGS == 'Power' else AUDIO
ENF_INDEXES = FREQ_60 if ENF == 60 else FREQ_50
SAMPLE_INDEXES = [i for i in RECORDING_INDEXES if i in ENF_INDEXES]


# =======================
# Load samples
# =======================
gnb = np.load('GNB_pred_P.npy', allow_pickle=True)
rf = np.load('RF_pred_P.npy', allow_pickle=True)
mlp = np.load('MLP_pred_P.npy', allow_pickle=True)
lr = np.load('LR_pred_P.npy', allow_pickle=True)

# Concatenate
concat = []
for i in range(100):
    if rf[i] is None:
        concat.append(None)
        continue
    concat.append(np.concatenate((gnb[i], rf[i], mlp[i], lr[i]), axis=1).flatten())
    #concat.append(np.concatenate((rf[i], mlp[i], lr[i]), axis=1).flatten())


# =======================
# Create training sets
# =======================
X = []
y = []

tempX = []
for i in SAMPLE_INDEXES:
    if TEST[i-1] == 'N':
        tempX.append(concat[i-1])
        continue
    X.append(concat[i-1])
    y.append(TEST[i-1])

fusion_model = MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, random_state=42)
fusion_model.fit(X, y)

y_pred = fusion_model.predict(X)
accuracy = accuracy_score(y, y_pred)
'''for i in range(len(X)):
    p = fusion_model.predict_proba([X[i]])[0]
    print(np.reshape(X[i], (6, 8)))
    print(y[i])
    # if max(p) < 0.98:
    #     print(max(p))
    # if max(p) < 0.9:
    #     print(np.reshape(X[i], (6, 8)))
    #     print(y[i])
'''


#asdf = fusion_model.predict_proba([tempX[0]])
y_pred = fusion_model.predict_proba(tempX)

print('kati')
