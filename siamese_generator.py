import cv2 as cv
import numpy as np
import keras
import os
import random
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

'''Data Members******************************************************************************************************'''
# directory: location of train / val image folders
# idxs: list of class indices for N classes in train / val sets
# labels: dictionary mapping class index to class id
# img_dict: dctionary mapping class id to image filename
'''******************************************************************************************************************'''

class SiamesePairGenerator(keras.utils.Sequence):

    def __init__(self, directory, idxs, labels, img_dict,
                 transform_gen = ImageDataGenerator(rescale=1./255),
                 dim = (224,224), n_channels = 3,
                 n_aug=20, batch_size = 32, shuffle = True):

        self.directory = directory
        self.idxs = idxs
        self.label_dict = labels
        self.img_dict = img_dict
        self.transform = transform_gen
        self.dim = dim
        self.channels = n_channels
        self.aug_factor = n_aug
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    # shuffle indices at the end of each epoch
    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.idxs)

    # return number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.idxs)/self.batch_size))

    # load image from directory
    def __load_image(self, fn):

        # print(os.path.join(self.directory))
        img = cv.imread(filename=os.path.join(self.directory,fn), flags=0)
        img = cv.resize(src=img, dsize=self.dim, dst=img)
        img = np.array(img,dtype=np.float32)/255
        img = np.stack((img, img, img), axis=-1)

        return img

    # generate a batch of image pairs for a set of indices
    def __data_generation(self, idxs):

        x = np.empty(shape=(self.batch_size, 2, 224,224, self.channels ),dtype=np.float32)
        y = np.empty(shape=(self.batch_size),dtype=int)

        for i, idx in enumerate(idxs):

            print("\nBatch Element: %d" %i)
            # import database image corresponding to idx
            class_id = self.label_dict[idx]
            img_fn = self.img_dict[class_id]
            img = self.__load_image(img_fn)
            print("\tImage 1: [%d,%s]: " %(idx,img_fn))

            # make a positive pair for even indices
            if i % 2 == 0:
                pair_img = img
                y[i,] = 1
                print("\tImage 2: [%d,%s]: " % (idx, img_fn))

            # make a negative pair for odd indices
            else:
                r = range(0,idx) + range(idx+1, len(self.idxs))
                random_idx = random.choice(r)

                rand_class_id = self.label_dict[random_idx]
                img_fn = self.img_dict[rand_class_id]
                pair_img = self.__load_image(img_fn)

                y[i,] = 0
                print("\tImage 2: [%d,%s]: " % (random_idx, img_fn))

            x[i,] = np.array([[img, pair_img]],dtype=np.float32)



        return x, y

    def __getitem__(self, idx):

        idxs = self.idxs[idx*self.batch_size:(idx+1)*self.batch_size]

        X, Y = self.__data_generation(idxs)

        temp = X[0][1]
        # print(temp)
        # plt.imshow(temp)
        # plt.show()

        # print(X.shape)
        # print(Y.shape)
        return X, Y


