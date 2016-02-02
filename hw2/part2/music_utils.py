import os
import numpy as np
import glob

# some global constants

MUSIC_DIR = "music/"

genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# read the fft representation and create X,y arrays for training

def read_fft(genre_list, base_dir):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, 'fft','*.fft.npy')
        file_list = glob.glob(genre_dir)
        for fname in file_list:
            fft_features = np.load(fname)
            X.append(fft_features[:1000])
            y.append(label)
    return np.array(X), np.array(y)

def read_ceps(genre_list, base_dir):
    X, y = [], []
    for label, genre in enumerate(genre_list):
        for fname in glob.glob(os.path.join(base_dir, genre,"ceps", "*.ceps.npy")):
            ceps = np.load(fname)
            num_ceps = len(ceps)
            X.append(np.mean(
                    ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0))
            y.append(label)
    return np.array(X), np.array(y)



