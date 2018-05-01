import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def localization_train_loader(tr_digits, noise=0.0):
    tr_digits = shuffle(tr_digits)
    for i in range((tr_digits.shape[0] + 16 - 1) // 16):
        sl = slice(i * 16, (i+1) * 16)
        X_b = tr_digits[sl] 

        X_loc = np.zeros((X_b.shape[0], 32, 32, 1))
        y_loc = np.zeros((X_b.shape[0], 4))
        for j in range(X_loc.shape[0]):
            rnd_y = np.random.randint(0, 24)
            rnd_x = np.random.randint(0, 24)
            X_loc[j][rnd_y:rnd_y+8, rnd_x:rnd_x+8] = X_b[j].reshape((8,8, 1)) / np.amax(X_b[j])
            X_loc[j] += np.random.normal(scale=noise, size=(X_loc[j].shape))
            y_loc[j] = np.asarray([rnd_x / 32., (rnd_x+8) / 32., rnd_y / 32., (rnd_y+8) / 32.])

        yield X_loc, y_loc

def make_valid_localization_data(tst_digits, noise=0.0):
    X_loc = np.zeros((tst_digits.shape[0], 32, 32, 1))
    y_loc = np.zeros((tst_digits.shape[0], 4))
    for i in range(tst_digits.shape[0]):
        rnd_y = np.random.randint(0, 24)
        rnd_x = np.random.randint(0, 24)
        X_loc[i][rnd_y:rnd_y+8, rnd_x:rnd_x+8] = tst_digits[i].reshape((8,8,1)) / np.amax(tst_digits[i])
        X_loc[i] += np.random.normal(scale=noise, size=(X_loc[i].shape))
        y_loc[i] = np.asarray([rnd_x / 32., (rnd_x+8) / 32., rnd_y / 32., (rnd_y+8) / 32.])

    return X_loc, y_loc

def loc_det_train_loader(tr_digits, tr_labels, noise=0.0):
    tr_digits, tr_labels = shuffle(tr_digits, tr_labels)
    for i in range((tr_digits.shape[0] + 16 - 1) // 16):
        sl = slice(i * 16, (i+1) * 16)
        X_b = tr_digits[sl] 
        y_class = tr_labels[sl]

        X_loc = np.zeros((X_b.shape[0], 32, 32, 1))
        y_loc = np.zeros((X_b.shape[0], 4))
        for j in range(X_loc.shape[0]):
            rnd_y = np.random.randint(0, 24)
            rnd_x = np.random.randint(0, 24)
            X_loc[j][rnd_y:rnd_y+8, rnd_x:rnd_x+8] = X_b[j].reshape((8,8, 1)) / np.amax(X_b[j])
            X_loc[j] += np.random.normal(scale=noise, size=(X_loc[j].shape))
            y_loc[j] = np.asarray([rnd_x / 32., (rnd_x+8) / 32., rnd_y / 32., (rnd_y+8) / 32.])

        yield X_loc, y_loc, y_class
