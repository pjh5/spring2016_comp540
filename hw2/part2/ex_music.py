import numpy as np
import music_utils
from one_vs_all import one_vs_allLogisticRegressor
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, classification_report

# some global constants

MUSIC_DIR = "music/"
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

# select the CEPS or FFT representation

X,y = music_utils.read_ceps(genres,MUSIC_DIR)
#X,y = music_utils.read_fft(genres,MUSIC_DIR)

# select a regularization parameter

reg = 1.0

# create a 1-vs-all classifier

ova_logreg = one_vs_allLogisticRegressor(np.arange(10))

#  divide X into train and test sets

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# train the K classifiers in 1-vs-all mode

ova_logreg.train(X_train,y_train,reg,'l2')

# predict on the set aside test set

ypred = ova_logreg.predict(X_test)
print confusion_matrix(y_test,ypred)
