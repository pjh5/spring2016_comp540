import numpy as np
from skimage import data
from skimage.transform import *

def augment(x):
  au = x
  for i in range(x.shape[0]):
    angle = np.random.ranf()*40 - 20
    c = np.random.ranf()*255
    resize = 1 + np.random.ranf()*0.5
    au[i] = rotate(au[i], angle, resize=False, mode='constant', cval=c)
    #au_t = rescale(au[i], resize)
    #b = au.shape[1] - 32
    #if b > 0:
    #  h = np.random.randint(0, b, 1)[0]
    #  w = np.random.randint(0, b, 1)[0]
    #else:
    #  h = 0 
    #  w = 0 
    #au[i] = au[i,:,h:32+h, w:32+w]
  return au
