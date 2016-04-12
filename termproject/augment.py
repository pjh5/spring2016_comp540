import numpy as np
from skimage import data
from skimage.transform import *
from skimage import io
from matplotlib import pyplot as plt
import sys
import utils2

ntrain = 50000
#data multiplication coefficient
m = 10

classes=["airplaine", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print ("Reading in the data...")
sys.stdout.flush()

TRAIN_DIR = "train/"
AUGMENTATION_DIR = "augmented/"

imgs = utils2.read_folder(TRAIN_DIR, 0, ntrain, flatten = False)
l = utils2.read_labels('trainLabels.csv', 0, ntrain)

labels = []
for i in range(l.shape[0]):
  labels.append(classes[l[i]])
  io.imsave(AUGMENTATION_DIR + str(i + 1) + ".png", imgs[i]/255)

print ("\nDone!")
sys.stdout.flush()

print ("Augmenting the data...")
sys.stdout.flush()
k = ntrain
for i in range(ntrain * m):
  k+=1
  sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (ntrain * m)))
  sys.stdout.flush()
  j = np.random.randint(0, ntrain, 1)[0]
  angle = np.random.randint(0, 80, 1)[0] - 40
  c = np.float(np.random.randint(0, 255, 1)[0])
  resize = 0.9 + np.float(np.random.randint(0, 50, 1)[0]) / 100
  #print angle
  #print resize
  au = rescale(imgs[j], resize)
  if au.shape[1] < 32:
    d = 32 - au.shape[1]
    au = np.lib.pad(au,((d,d),(d,d),(0,0)),'constant',constant_values=c)
  c = np.float(np.random.randint(0, 255, 1)[0])
  au = rotate(au, angle, resize=False, mode='constant', cval=c) / 255
  b = au.shape[1] - 32
  if b > 0:
    h = np.random.randint(0, b, 1)[0]
    w = np.random.randint(0, b, 1)[0]
  else:
    h = 0 
    w = 0 
  au = au[h:32+h, w:32+w]
  #print au.shape
  io.imsave(AUGMENTATION_DIR + str(k) + ".png", au)
  labels.append(classes[l[j]])

ids = np.array(np.arange(1, len(labels) + 1)).reshape((1, len(labels)))
labels = np.array(labels).reshape((1, len(labels)))

l = np.zeros(ids.size, dtype=[('var1', int), ('var2', 'S21')])
l['var1'] = ids
l['var2'] = labels

np.savetxt('augmented.csv', l, fmt="%d,%s", delimiter=',')
print "\nDone!\n"