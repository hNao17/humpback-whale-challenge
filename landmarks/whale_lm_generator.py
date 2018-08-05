import os
import math

import keras
import numpy as np
from skimage import io, transform, exposure
from skimage.filters import gaussian

class WhaleLMGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, data, n_landmarks, batch_size=32, dim=(224,224), n_channels=1,
             shuffle=True, transforms=[], is_test=False, seed=None):
        self.root_dir = root_dir
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.n_landmarks = n_landmarks
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.im_names = [k for k in self.data]
        self.indexes = np.arange(len(self.data))
        self.transforms = transforms
        self.is_test = is_test
        self.on_epoch_end()
        
        if seed:
            np.random.seed(seed)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __read_image(self, im_name):
        im_path = os.path.join(self.root_dir, im_name)
        img = io.imread(im_path, as_grey=True)
        img = transform.resize(img, self.dim)
        
        p2, p98 = np.percentile(img, (2, 98))
        img = exposure.rescale_intensity(img, in_range=(p2, p98))
        
        return img[:, :, np.newaxis]
    
    def __rotate(self, X, lms):
        d = np.random.uniform(-45, 45)
        
        h1, w1 = X.shape[:2]
        X = transform.rotate(X, -d, resize=True, mode='edge')
        h2, w2 = X.shape[:2]
        
        scale = (h2 / h1, w2 / w1)
        
        d = math.radians(d)
        
        R = np.array([[math.cos(d), -math.sin(d)], [math.sin(d), math.cos(d)]])
        
        lms = np.dot(lms - 0.5, R.T)
        lms[:,0] /= scale[0]
        lms[:,1] /= scale[1]
        lms += 0.5
        
        X = transform.resize(X, self.dim)
        
        return X, lms
    
    def __rescale(self, X, lms):
        scale_factor = np.random.uniform(0.5, 1.0)
        
        new_size = int(self.dim[0] / scale_factor)
        X = np.pad(X[:,:,0], pad_width=(new_size-self.dim[0]) // 2, mode='constant', constant_values=1.)
        X = transform.resize(X, self.dim)
        X = X[:,:,np.newaxis]
        
        lms -= 0.5
        lms *= scale_factor
        lms += 0.5
        
        return X, lms
    
    def __horiz_flip(self, X, lms):
        if np.random.rand() > 0.5:
            X = np.flip(X, axis=1)
            lms[:,0] = 1. - lms[:,0]
            tmp = np.copy(lms[0,:])
            lms[0,:] = lms[2,:]
            lms[2,:] = tmp

        return X, lms
    
    def __data_generation(self, idxs):
        """Generates data containing batch_size 
        samples' # X : (n_samples, *dim, n_channels)."""
        
        X = np.empty((len(idxs), *self.dim, self.n_channels))
        Y = np.zeros((len(idxs), self.n_landmarks * 2))
        
        for i, idx in enumerate(idxs):
            im_name = self.im_names[idx]
            X[i,] = self.__read_image(im_name)
            k = len(self.data[im_name])
            lms = np.array([[p['x'], p['y']] for p in self.data[im_name]])
            
            if 'horizontal_flip' in self.transforms:
                X[i,], lms = self.__horiz_flip(X[i,], lms)
            if 'rescale' in self.transforms:
                X[i,], lms = self.__rescale(X[i,], lms)
            if 'rotate' in self.transforms:
                X[i,], lms = self.__rotate(X[i,], lms)
            
            Y[i,:k*2] = lms.flatten()
        
        return X, Y
    
    def __len__(self):
        'Denotes the number of batches per epoch.'
        return len(self.data) // self.batch_size
    
    def __getitem__(self, idx):
        'Generate one batch of data.'
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X, Y = self.__data_generation(idxs)
        
        return X, Y