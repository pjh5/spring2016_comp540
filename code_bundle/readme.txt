There are two folders with code:

1. not NNs - this is a folder with our code, that doesn't use neural networks.
2. NNs - code that uses neural networks.

------------------------------- Not NNs --------------------------------------

This is a folder with approaches without neural networks.

All these files require subfolders with raw .png pictures. By default they are
named 'train/' for training and 'test/' for testing respectively.
There are additional comments inside these files.

There are three approaches presented there:

1. KNN. This classification can be run by script 'main_knn.py'. With 50000 
training images this will take hours to run.

2. HOG. This classification can be run by script 'main_HOG.py'.

3. HOG for every RGB channel of the picture. This can be run by script 
'main_HOG3_softmax.py' with SoftMax classifier and 'main_HOG3_svm.py' with SVM
classifier. BOTH THESE SCRIPTS WILL REQUIRE AT LEAST 10 GB of RAM!

--------------------------------- NNs ----------------------------------------

This is a folder with neural networks.

There are two subfolders there - 'caffe' and 'our code'

-------------------------------- Caffe ---------------------------------------

1. Inside the folder caffe you will find all the files required for 
classification with working installation of Caffe. By default these files need
to be placed in 'examples/cifar10' folder. You can request snapshot files (300
MB, they are too big to be placed on GitHub for free), that already have 
trained model, because training will take around 12 hours. Training is made by 
running file 'train.sh', predicting on the test set - by running 'predict.py' 
script. Prediction also takes several hours.

By default all the scripts were run from Caffe's root folder. Commands are as
follows:

'./examples/cifar10/train.sh'  -  this command will run training of our model.

'./examples/cifar10/run_trained.sh'  -  this command will run trained model for
10 000 more iterations.

'python ./examples/cifar10/predict.py'  -  this command will run prediction on 
the test pictures (.png pictures by default placed in '/home/<username>/test/'
folder)

------------------------------ Our code --------------------------------------

2. Inside the folder 'our code' you will find files that will run neural
networks, written by us. This is the code from our assignments and also from
the assignments by Karpathy's course in Stanford (dropout).

'main_3net.py' - runs 3-layer fully connected net.
