import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import random

class MNIST(Dataset):
    def __init__(self, data, transform=None, augmented=None):


        self.data = data
        self.img = self.data['input_images']
        self.labels = self.data['output_labels']
        self.real_img = self.data['input_images']
        self.value = augmented
        self.transform = transform

        if augmented != None:
            ### Set real images
            print('Augmentation process with noise value {}'.format(self.value))
            self.data['real_images'] = self.data['input_images']

            ### Augment Data using white noise
            for noise in self.value:
                # Separate pre and post noise data
                real_data = self.data['real_images']
                noisy_data = self.data['input_images'] + np.random.rand(self.data['input_images'].shape[0], self.data['input_images'].shape[1])*noise
                # update data
                self.data['input_images'] = np.concatenate((self.data['input_images'], noisy_data))
                self.data['real_images'] = np.concatenate((self.data['real_images'], real_data))
                self.data['output_labels'] = np.concatenate((self.data['output_labels'], self.data['output_labels']))

            ### Update dataset
            print('Size of augmented data: {}'.format(data['input_images'].shape[0]))
            self.img = self.data['input_images']
            self.labels = self.data['output_labels']
            self.real_img = self.data['real_images']

    def __getitem__(self, idx):
        # Create sample
        sample = {'img': torch.tensor(np.reshape(self.img[idx], (1, 28, 28))), 'real_img': torch.tensor(np.reshape(self.real_img[idx], (1, 28, 28))), 'label': torch.tensor(self.labels[idx])}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        if self.img.shape[0] != self.labels.shape[0]:
           print('Different number of img and labels. Expected same number, got {} and {}'.format(self.img.shape[0], self.labels.shape[0]))
           print('Returning None')
           return None
        return self.img.shape[0]


def augmentation(data, values):
    for noise in [0.1, 0.2]:
        X1 = data['input_images'] + np.random.rand(data['input_images'].shape[0], data['input_images'].shape[1])*noise
        data['input_images'] = np.concatenate((data['input_images'], X1))
        data['output_labels'] = np.concatenate((data['output_labels'], data['output_labels']))
    return data

def main():
    # Load Data
    print('Trying to load MNIST.mat for demo. Please make sure to have it in the folder')
    data = spio.loadmat('MNIST.mat')
    print('Size of raw data: {}'.format(data['input_images'].shape[0]))

    # Augment Data
    #augmented_data = augmentation(data, values=[0.1, 0.2, 0.3])
    #print('Size of raw data: {}'.format(augmented_data['input_images'].shape[0]))

    # Create Dataset
    dataset = MNIST(data, augmented=[0.1, 0.3])
    # Set Params
    batch_size = 128
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

    print('Shape of images:', end = '\t')
    for i in train_dataloader:
        print(i['img'].shape)
        break

if __name__ == '__main__':
    main()
