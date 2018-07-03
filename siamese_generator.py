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
# img_dict: dictionary mapping class id to image filename
'''******************************************************************************************************************'''

class SiamesePairGenerator(keras.utils.Sequence):

    def __init__(self, directory, idxs, labels, img_dict,
                 transform_gen = None,
                 dim = (224,224), n_channels = 3,
                 n_aug=1, batch_size = 32, shuffle = True):

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
        # print(self.idxs[0])

    # return number of batches per epoch
    def __len__(self):
        #return int(np.floor(len(self.idxs)/self.batch_size))
        return int(np.floor(len(self.idxs)*self.aug_factor/self.batch_size))

    # load image from directory
    def __load_image(self, fn):

        # print(os.path.join(self.directory))
        img = cv.imread(filename=os.path.join(self.directory,fn), flags=0)
        img = cv.resize(src=img, dsize=self.dim, dst=img)
        img = np.array(img,dtype=np.float32)/255
        img = np.stack((img, img, img), axis=-1)

        return img

    # apply transforms to an input image
    def __augment_image(self,fn):

        img = self.__load_image(fn)
        img = img.reshape((1,) + img.shape)

        # augment original image once
        aug_counter = 0
        for batch in self.transform.flow(x=img, batch_size=1):

            aug_counter += 1
            if aug_counter > 0:
                return batch[0]


    '''1 to N pair generation'''

    # generate a batch of image pairs for a set of indices
    def __data_generation(self, idx):

        x = np.empty(shape=(self.batch_size, 2, 224,224, self.channels ),dtype=np.float32)
        y = np.empty(shape=(self.batch_size),dtype=int)

        # import database image corresponding to idx
        start_idx = np.min(self.idxs)
        print("Start Idx: %d " %start_idx)
        class_idx = self.idxs[idx]
        class_id = self.label_dict[class_idx]
        db_fn = self.img_dict[class_id]
        # img_db = self.__load_image(db_fn)

        for i in range(0,self.aug_factor):

            print("\nBatch Element: %d" % i)
            print("\tImage 1: [%d,%s]: " % (class_idx, db_fn))

            if self.transform is None:
                img_db = self.__load_image(db_fn)
            else:
                img_db = self.__augment_image(db_fn)

            # make a positive pair for even indices
            if i % 2 == 0:
                if self.transform is None:
                    pair_img = img_db
                else:
                    pair_img = self.__augment_image(db_fn)
                y[i,] = 1
                print("\tImage 2: [%d,%s]: " % (class_idx, db_fn))

            # make a negative pair for odd indices
            else:
                r = range(start_idx, class_idx) + range(class_idx + 1, len(self.idxs))
                random_idx = random.choice(r)

                rand_class_id = self.label_dict[random_idx]
                rand_fn = self.img_dict[rand_class_id]

                if self.transform is None:
                    pair_img = self.__load_image(rand_fn)
                else:
                    pair_img = self.__augment_image(rand_fn)

                y[i,] = 0
                print("\tImage 2: [%d,%s]: " % (random_idx, rand_fn))

            x[i,] = np.array([[img_db, pair_img]], dtype=np.float32)


        return x, y


    # return batch of image data and labels
    def __getitem__(self, idx):

        print("Current database idx: %d" %idx)
        # idxs = self.idxs[idx*self.aug_factor:(idx+1)*self.aug_factor]

        X, Y = self.__data_generation(idx)

        return X, Y

    '''1 to 1 pair generation'''
    '''
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

    # return batch of image data and labels 
    def __getitem__(self, idx):

        print("Current database idx: %d" %idx)
        idxs = self.idxs[idx*self.batch_size:(idx+1)*self.batch_size]

        X, Y = self.__data_generation(idxs)

        return X, Y
    '''

