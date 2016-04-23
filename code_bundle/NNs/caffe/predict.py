import matplotlib 
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import caffe
caffe.set_mode_gpu()

classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

ids = []
labels = []

# this loop reads 300 000 images from the test folder
ntest = 300000

fname = np.chararray((ntest), itemsize=255)
imgs = np.zeros((ntest, 32, 32, 3))
for i in range(ntest):
	sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (ntest)))
	sys.stdout.flush()
	# here is the path to the test set
	fname[i]='/home/ubuntu/test/' + str(i+1) + '.png'
	#print fname[i]
	#sys.stdout.flush()
	imgs[i] = caffe.io.load_image(fname[i])

imgs *= 255
print "Done!"
sys.stdout.flush()

# model files
MODEL_FILE = '/home/ubuntu/caffe/examples/cifar10/cifar10_my.prototxt'
PRETRAINED = '/home/ubuntu/caffe/examples/cifar10/cifar10_my_iter_40000.caffemodel'

# read mean image
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/ubuntu/caffe/examples/cifar10/mean.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))

# create model
net = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=(32, 32), mean=arr[0])

# this loop predicts on the test dataset in batches of 1000 pictures
batch = 1000
nbatches = np.int(ntest / batch)
print "Predicting on the test dataset...\n"
for i in range(nbatches):
	sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (nbatches)))
	sys.stdout.flush()
	prediction = net.predict(imgs[i*batch:(i+1)*batch])
	cat = np.argmax(prediction, axis=1)
	for k in range(prediction.shape[0]):
		labels.append(classes[np.int(cat[k])])
		ids.append(i * batch + k + 1)

ids = np.array(ids).reshape((len(ids), 1))
labels = np.array(labels).reshape((len(labels), 1))


l = np.concatenate((ids, labels), axis=1)

# write output file
out = open('testLabels.csv', 'w')
out.write('%s' % 'id,label')
out.write('\n')

for row in l:
	out.write('%s,' % row[0])
	out.write('%s' % row[1])
	out.write('\n')
out.close()

print ("\nDone!")