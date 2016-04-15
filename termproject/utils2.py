import os
import numpy as np
from scipy import misc
from skimage.feature import hog
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import disk
from skimage import exposure
import sys
#import cv2
from matplotlib import pyplot as plt
from knn import KNNClassifier
from sklearn.metrics import confusion_matrix


def read_folder(base_dir, min=0, max=-1, flatten=True):

	#read only max files if stated
	l = os.listdir(base_dir)
	
	#if max is not given or wrong - then read all the folder
	if(max == -1 or max > len(l)):
		max = len(l)
	
	if flatten:
		img = np.zeros((max - min, 32, 32))
	else:
		img = np.zeros((max - min, 32, 32, 3))
	for i in range(max - min):
		#print ("Reading ", str(base_dir + str(min + i + 1) + ".png"))
		img[i] = misc.imread(str(base_dir + str(min + i + 1) + ".png"), flatten = flatten)
		sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (max - min)))
		sys.stdout.flush()
	return img
	
def cv_read_folder(base_dir, min=0, max=-1):

	#read only max files if stated
	l = os.listdir(base_dir)
	
	#if max is not given or wrong - then read all the folder
	if(max == -1 or max > len(l)):
		max = len(l)
	
	imgs = []
	
	for i in range(max - min):
		imgs.append(cv2.imread(str(base_dir + str(min + i + 1) + ".png"), 1))
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return imgs
	
def getHOG(imgs, ori=8, ppc=(4,4), cpb=(4,4), vis=True):
	#determine the shape of the output
	if vis:
		fd, im = hog(imgs[0,:], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=vis)
		imgs2 = imgs
	else:
		fd = hog(imgs[0,:], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=vis)
	
	hogs = np.zeros((imgs.shape[0], fd.shape[0]))
	#HOG
	for i in range(imgs.shape[0]):
		#zimgs[i,:] = exposure.equalize_hist(imgs[i,:])
		#imgs[i,:] = rank.equalize(imgs[i,:]/255,selem=disk(0))
		#plt.imshow(imgs[i,:]),plt.show()
		if vis:
			hogs[i,:], imgs2[i] = hog(imgs[i,:], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=vis)
		else:
			hogs[i,:] = hog(imgs[i,:], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=vis)
		sys.stdout.write("\rIteration {0}/{1}".format((i + 1), imgs.shape[0]))
		sys.stdout.flush()
	mean = np.mean(hogs, axis = 0)
	hogs -= mean
	
	if vis:
		return hogs, imgs2
	else:
		return hogs
	
def getHOG3(imgs, ori=8, ppc=(4,4), cpb=(4,4)):
	#determine the shape of the output
	fd = hog(imgs[0,:,:,0], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=False)
	#print fd.shape
	hogs = np.zeros((imgs.shape[0], fd.shape[0] * 3))
	#HOG
	for i in range(imgs.shape[0]):
		#zimgs[i,:] = exposure.equalize_hist(imgs[i,:])
		#imgs[i,:] = rank.equalize(imgs[i,:]/255,selem=disk(0))
		#plt.imshow(imgs[i,:]),plt.show()
		hogs[i,0:fd.shape[0]] = hog(imgs[i,:,:,0], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=False)
		hogs[i,fd.shape[0]:(2 * fd.shape[0])] = hog(imgs[i,:,:,1], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=False)
		hogs[i,2 * fd.shape[0]:(3 * fd.shape[0])] = hog(imgs[i,:,:,2], orientations=ori, pixels_per_cell=ppc, cells_per_block=cpb, visualise=False)
		sys.stdout.write("\rIteration {0}/{1}".format((i + 1), imgs.shape[0]))
		sys.stdout.flush()
		
	mean = np.mean(hogs, axis = 0)
	hogs -= mean
	return hogs
		
def read_labels(file, min=0, max=50000, classes=["airplaine", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]):
	
	labels = np.genfromtxt(file, delimiter = ',', dtype = 'str', skip_header = 1).astype(str)
	labels = np.genfromtxt(file, delimiter = ',', dtype = 'str', skip_header = 1).astype(str)
	y = np.zeros(max - min)
	
	for i in range(max - min):
		for j in range(0, 10):
			if labels[i,1] == classes[j]:
				y[i] = j
	y = y.astype(int)
	
	return y

def getORB(imgs):
	imgs2 = imgs
	orb = cv2.ORB(nfeatures=10,scaleFactor=1.2,nlevels=1,edgeThreshold=0,patchSize=2)
	d = np.zeros((len(imgs),10,32))
	for i in range(len(imgs)):
		kp, des = orb.detectAndCompute(imgs[i], None)
		imgs2[i] = cv2.drawKeypoints(imgs[i],kp,color=(0,0,255), flags=0)
		if des is None:
			print "No points found for img #",i,"!"
			sys.stdout.flush()
		else:
			for j in range(des.shape[0]):
				d[i,j] = des[j]
	return d, imgs2
	
def showORB(imgs, y_train):
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes)
	samples_per_class = 7
	for y, cls in enumerate(classes):
		idxs = np.flatnonzero(y_train == y)
		idxs = np.random.choice(idxs, samples_per_class, replace=False)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(samples_per_class, num_classes, plt_idx)
			plt.imshow(imgs[idx].astype('uint8'))
			plt.axis('off')
			if i == 0:
				plt.title(cls)
	plt.savefig('orb_samples.pdf')
	plt.close()

def showHOG(imgs, y_train):
	classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes)
	samples_per_class = 7
	for y, cls in enumerate(classes):
		idxs = np.flatnonzero(y_train == y)
		idxs = np.random.choice(idxs, samples_per_class, replace=False)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(samples_per_class, num_classes, plt_idx)
			plt.imshow(imgs[idx], cmap=plt.cm.gray)
			plt.axis('off')
			if i == 0:
				plt.title(cls)
	plt.savefig('hog_samples.pdf')
	plt.close()
	
def visualizeHOGTheta(theta):
	directions = np.zeros((8,32,32))
	for i in range(8):
		name = "grad" + str(i) + ".jpg"
		directions[i] = misc.imread(name, flatten = True)
	
	classes = ['grad','plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	num_classes = len(classes) + 1
	orientations = 8
	for y, cls in enumerate(classes):
		idxs = np.arange(orientations)
		for i, idx in enumerate(idxs):
			plt_idx = i * num_classes + y + 1
			plt.subplot(orientations, num_classes, plt_idx)
			if y == 0:
				plt.imshow(directions[i], cmap=plt.cm.gray)
			else:
				plt.imshow(theta[:,:,i,y - 1], cmap=plt.cm.gray)
			plt.axis('off')
			if i == 0:
				plt.title(cls)
	plt.savefig('theta_HOG.pdf')
	plt.close()

def getBestRegAndLearnSoftMax(classifier, X_train, y_train, X_val, y_val, learning_rates = [3e0], regularization_strengths = [5e-2], print_train=True, print_val=True):
	acc_train = np.zeros((len(learning_rates), len(regularization_strengths)))
	acc_val = np.zeros((len(learning_rates), len(regularization_strengths)))
	for i in range(0, len(learning_rates)):
		for j in range(0, len(regularization_strengths)):
			print "Training SoftMax with LR #",str(i + 1),"/",len(learning_rates),", RS #",str(j + 1),"/",len(regularization_strengths)
			sys.stdout.flush()
			classifier.train(X_train, y_train, reg=regularization_strengths[j], learning_rate=learning_rates[i])
			acc_train[i, j] = np.mean(classifier.predict(X_train) == y_train)
			acc_val[i, j] = np.mean(classifier.predict(X_val) == y_val)
	
	if print_train:
		print "LR:",learning_rates
		print "RS:",regularization_strengths
		print "Traning accuracy:"
		print acc_train
	if print_val:
		print "Validation accuracy:"
		print acc_val
	sys.stdout.flush()

	best_val = np.max(acc_val)
	best_rate, best_reg = np.where(acc_val==np.amax(acc_val))
	return (best_rate[0], best_reg[0]), classifier

def getBestRegOVA(classifier, X_train, y_train, X_val, y_val, regularization_strengths = [5e-2], pen='l1', print_train=True, print_val=True):
	acc_train = np.zeros((1, len(regularization_strengths)))
	acc_val = np.zeros((1, len(regularization_strengths)))
	for j in range(0, len(regularization_strengths)):
		classifier.train(X_train, y_train, reg=regularization_strengths[j], penalty=pen)
		acc_train[0, j] = np.mean(classifier.predict(X_train) == y_train)
		acc_val[0, j] = np.mean(classifier.predict(X_val) == y_val)
	
	if print_train:
		print (acc_train)
	if print_val:
		print (acc_val)
	
	best_val = np.max(acc_val)
	best_rate, best_reg = np.where(acc_val==np.amax(acc_val))
	return (best_rate[0], best_reg[0]), classifier
	
def getBestK(X_train, y_train, X_val, y_val, nns = [30], print_train=True, print_val=True):
	acc_train = np.zeros((1, len(nns)))
	acc_val = np.zeros((1, len(nns)))
	for j in range(0, len(nns)):
		print j
		sys.stdout.flush()
		knn = KNNClassifier(nns[j])
		knn.train(X_train, y_train)
		#acc_train[0, j] = np.mean(knn.predict(X_train) == y_train)
		print acc_train[0, j]
		sys.stdout.flush()
		y_pred = knn.predict(X_val)
		acc_val[0, j] = np.mean(y_pred == y_val)
		print acc_val[0, j]
		sys.stdout.flush()
		print "Confusion matrix:"
		print confusion_matrix(y_pred, y_val)
	
	if print_train:
		print (acc_train)
	if print_val:
		print (acc_val)
	
	best_val = np.max(acc_val)
	best_rate, best_reg = np.where(acc_val==np.amax(acc_val))
	return (best_rate[0], best_reg[0]), knn
	
def sigmoid (z):
    sig = np.zeros(z.shape)
    # Your code here
    # 1 line expected
    sig = 1 / (1 + 1 / np.exp(z))
    # End your ode

    return sig