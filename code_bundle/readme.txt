There are two folders with code:

1. not NNs - this is a folder with our code, that doesn't use neural networks.
2. NNs - code that uses neural networks.

Not NNs.

All these files require subfolders with raw .png pictures. By default they are
named 'train/' for training and 'test/' for testing respectively.
There are additional comments inside these files.

There are four approaches presented there:
1. KNN. This classification can be run by script 'main_knn.py'. With 50000 
training images this will take hours to run.
2. HOG. This classification can be run by script 'main_HOG.py'.
3. HOG for every RGB channel of the picture. This can be run by script 
'main_HOG3_softmax.py' with SoftMax classifier and 'main_HOG3_svm.py' with SVM
classifier. BOTH THESE SCRIPTS WILL REQUIRE AT LEAST 10 GB of RAM!

NNs

There are two subfolders there - 'caffe' and 'our code'

1. Inside the folder caffe you will find all the files required for 
classification with working installation of Caffe. By default these files need
to be placed in 'examples/cifar10' folder. You can request snapshot files (300
MB), that already have trained model, because training will take around 10
hours. Training is made by running file 'train.sh', predicting on the test set
 - by running 'predict.py' script. Prediction also takes several hours.

2. Inside the folder 'our code' you will find files that will run neural
networks, written by us. This is the code from our assignments and also from
the assignments by Karpathy's course in Stanford.

'main_3net.py' - runs 3-layer fully connected net.
