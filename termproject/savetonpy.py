import numpy as np
import sys
import utils
TEST_DIR = "test/"

batch = 50
for j in range(0, 6):
	print "\nPart ", j + 1, "of 6"
	sys.stdout.flush()
	test_imgs = utils.read_folder(TEST_DIR, j * batch, (j + 1) * batch, flatten = False)
	name = "tbatch_" + str(j)
	#np.save(name, np.around(test_imgs, 2))
	np.savez_compressed(name, test_imgs)