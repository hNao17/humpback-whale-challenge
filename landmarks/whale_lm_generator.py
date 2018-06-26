import os

import keras
import numpy as np
from skimage import io, transform
from skimage.filters import gaussian

class WhaleLMGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, data, n_landmarks, batch_size=32, dim=(224,224), n_channels=1,
             shuffle=True, transforms=None):
        self.root_dir = root_dir
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.n_landmarks = n_landmarks
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.im_names = [k for k in self.data]
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __read_image(self, im_name):
        im_path = os.path.join(self.root_dir, im_name)
        image = io.imread(im_path, as_grey=True)
        image = transform.resize(image, self.dim)
        return image[:, :, np.newaxis]
    
    def __landmarks2mask(self, landmarks):
        h, w = self.dim
        k = len(landmarks)

        mask = np.zeros((w, h, self.n_landmarks), dtype=np.float32)
        
        for i in range(k):
            p = landmarks[i]
            mask[int(p['y'] * w), int(p['x'] * h), i] = 1.
            mask[:,:,i] = gaussian(image=mask[:,:,i], sigma=self.sigma)
        
        return mask
    
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
            Y[i, :k*2] = lms.flatten()
        
        return X, Y
    
    def __len__(self):
        'Denotes the number of batches per epoch.'
        return len(self.data) // self.batch_size
    
    def __getitem__(self, idx):
        'Generate one batch of data.'
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X, Y = self.__data_generation(idxs)
        
        return X, Y