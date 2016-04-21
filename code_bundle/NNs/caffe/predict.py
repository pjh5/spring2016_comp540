import matplotlib 
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/caffe/python/'  # this file is expected to be in {caffe_root}/examples
#import sys
#sys.path.insert(0, caffe_root + 'python')
classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
import caffe
caffe.set_mode_gpu()
ids = []
labels = []
fname = np.chararray((300000), itemsize=255)
imgs = np.zeros((300000, 32, 32, 3))
for i in range(300000):
	sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (300000)))
	sys.stdout.flush()
	fname[i]='/home/ubuntu/test/' + str(i+1) + '.png'
	#print fname[i]
	#sys.stdout.flush()
	imgs[i] = caffe.io.load_image(fname[i])
imgs*=255
print "Done!"
sys.stdout.flush()
MODEL_FILE = '/home/ubuntu/caffe/examples/cifar10/cifar10_my.prototxt'
PRETRAINED = '/home/ubuntu/caffe/examples/cifar10/cifar10_my_iter_40000.caffemodel'
IMAGE_FILE = '/home/ubuntu/test/183111.png'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/ubuntu/caffe/examples/cifar10/mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(32, 32), mean=arr[0])
#net.set_raw_scale('data', 255)
for i in range(300):
	sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (300)))
	sys.stdout.flush()
	prediction = net.predict(imgs[i*1000:(i+1)*1000])
	cat = np.argmax(prediction, axis=1)
	for k in range(prediction.shape[0]):
		labels.append(classes[np.int(cat[k])])
		ids.append(i * 1000 + k + 1)

ids = np.array(ids).reshape((len(ids), 1))
labels = np.array(labels).reshape((len(labels), 1))

l = np.concatenate((ids, labels), axis=1)

out = open('testLabels.csv', 'w')
for row in l:
    for column in row:
        out.write('%s,' % column)
    out.write('\n')
out.close()

print ("Done!")