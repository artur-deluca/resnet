import numpy as np
import os
import pickle
import requests
import tarfile


##TODO: implement input treatment used in the paper (randomization and per pixel mean subtraction)

class CIFAR:
    
    def __init__(self, data_dir=None, class_num=10, batch_size=124):
        """
        Arguments:
            data_dir: str, default None
                Directory containing datasets for training and testing. 
                If None it will download the dataset
            class_num: int, default 10
                Number of classes in the dependent variable (e.g CIFAR-10 --> 10)
            batch_size = int, default 32
                Size of batches to train network
        """
        if data_dir is None:
            if not hasattr(self, 'data_dir'):
                self.dataset_fetcher()
        else:
            assert type(data_dir) is str
            self.data_dir = data_dir

        # add attributes from funct
        self.batch_size = batch_size
        self.class_num = class_num

        # get files for training and validation
        train_files = [name for name in os.listdir(self.data_dir) if 'data_batch' in name]
        validation_files = [name for name in os.listdir(self.data_dir) if 'test_batch' in name]

        # read and merge files
        self.train_X, self.train_Y = [], []
        self.validation_X, self.validation_Y = [], []

        for file in train_files:
            X, Y = self._unpickle_CIFAR(os.path.join(self.data_dir, file), X_field='data', Y_field='labels')
            # append reshaped images
            self.train_X.append(X)
            # append labels
            self.train_Y.append(Y)
        
        self.train_X = np.concatenate(self.train_X, 0)
        self.train_Y = np.concatenate(self.train_Y, 0)

        for file in validation_files:
            X, Y = self._unpickle_CIFAR(os.path.join(self.data_dir, file), X_field='data', Y_field='labels')
            # append reshaped images
            self.validation_X.append(X)
            # append labels
            self.validation_Y.append(Y)
        
        self.validation_X = np.concatenate(self.validation_X, 0)
        self.validation_Y = np.concatenate(self.validation_Y, 0)

        # shuffle the training set
        rand_index = np.random.permutation(self.train_X.shape[0])
        self.train_X = self.train_X[rand_index]
        self.train_Y = self.train_Y[rand_index]

        # one hot encode the dependent variable
        self.train_Y = np.array([self._encode_one_hot(i, self.class_num) for i in self.train_Y])
        self.validation_Y = np.array([self._encode_one_hot(i, self.class_num) for i in self.validation_Y])

        self.count_train, self.count_validation  = 0, 0

        # split batches
        self.train_data_size = len(self.train_Y)
        self.train_num_batches = int(np.ceil(1.0 * self.train_data_size / self.batch_size))

        self.validation_data_size = len(self.validation_Y)
        self.validation_num_batches = int(np.ceil(1.0 * self.validation_data_size / self.batch_size))

    def dataset_fetcher(self, url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', chunks=15, dir='./dataset'):
        """Connects with the CIFAR-10 dataset url, downloads it in chunks and stores it in a pickle file
        Arguments:
            url: str, default CIFAR-url
            chunks: int, default 15
            dir: str, default './dataset'
        """
        file = self._download_file(url, chunks=chunks)
        self.data_dir = self._untar_CIFAR(file)
        
    def next_batch_train(self):
        
        next_X = self.train_X[self.count_train:self.count_train + self.batch_size]
        next_Y = self.train_Y[self.count_train:self.count_train + self.batch_size]
        
        self.count_train = (self.count_train + self.batch_size) % self.train_data_size
        
        return next_X, next_Y

    def next_batch_validation(self):
        
        next_X = self.validation_X[self.count_validation:self.count_validation + self.batch_size]
        next_Y = self.validation_Y[self.count_validation:self.count_validation + self.batch_size]
        
        self.count_validation = (self.count_validation + self.batch_size) % self.validation_data_size
        
        return next_X, next_Y

    def set_counter(self, value, which_set='validation'):
        if which_set == 'train':
            self.count_train = value
        elif which_set == 'validation':
            self.count_validation = value
    
    @staticmethod
    def _unpickle_CIFAR(file_dir, X_field='data', Y_field='labels'):
        """Unpickle the CIFAR dataset
        Arguments:
            file_dir: str
                Directory containing file
            X_field: str, default 'data'
                Name of the field in the file that contains X
            Y_field: str, default 'labels'
                Name of the field in the file that contains Y
        Returns:
            X: np.array, Y: np.array
        
        """
        # unpickle files
        with open(file_dir, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            
            # converting keys from bytes to strings  
            datadict = dict(zip(list(map(lambda x: x.decode('ascii'), datadict.keys())), datadict.values()))

            X = datadict[X_field].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(datadict['labels'])
        return X, Y
    
    @staticmethod
    def _download_file(url, chunks=15):
        local_filename = url.split('/')[-1]
        file_size = int(requests.head(url).headers['Content-Length'])
        
        r = requests.get(url, stream=True)
        with open(local_filename, 'wb') as f:
            i = 1
            for chunk in r.iter_content(chunk_size=file_size//chunks):
                print('chunk number: {} of {}'.format(i, chunks), end='\r') 
                if chunk:
                    f.write(chunk)
                    i+=1
        print()
        return local_filename
    
    @staticmethod
    def _untar_CIFAR(file, dir='./dataset', delete=True):
        """Extracts file into directory
        Arguments:
            file: str
                path to file
            dir: str, default './dataset'
                path to store extracted files
            delete: bool, default True
                delete .tar file
        """
        tar = tarfile.open(file)
        tar.extractall('./dataset')
        tar.close()
        if delete:
            os.remove(file)
        return os.path.join(dir, 'cifar-10-batches-py')
    
    @staticmethod
    def _encode_one_hot(x, classes):
        """One hot encode the dataset
        Arguments:
            x: numpy ndarray
                Data to encode
            classes: int
                Number of classes in dataset
        Returns:
            encoded data
        """
        one_hot = np.zeros(classes)
        one_hot[x] = 1
        return one_hot
    
    
    