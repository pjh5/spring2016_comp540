TEST_DIR = "../../../termproject/test/"
classes=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print "Making the final prediction..."
batch = 50000
ids = []
labels = []
for j in range(0, 6):
    print "\nPart ", j + 1, " of 6"
    sys.stdout.flush()
    test_imgs = utils2.read_folder(TEST_DIR, j * batch, (j + 1) * batch, flatten=False)
    print "\n"
    prediction = np.zeros(50000)
    for i in range(500):
        sys.stdout.write("\rIteration {0}/{1}".format((i + 1), (500)))
        sys.stdout.flush()
        prediction[i*100:(i+1)*100] = np.argmax(model.loss(test_imgs[i*100:(i+1)*100].swapaxes(1,3)), axis=1)
    for k in range(prediction.shape[0]):
        labels.append(classes[np.int(prediction[k])])
        ids.append(j * batch + k + 1)

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