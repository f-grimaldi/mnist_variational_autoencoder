import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import torch
from torch.autograd import Variable

import os
import argparse
import sys
import json

import VAE

parser = argparse.ArgumentParser(description='Variational Auto Encoder with MNIST dataset')

### ARGS
"""parser.add_argument('--model_dir', type=str, default='pre_trained_model',
                    help='directory where are stored the model variable to use')
parser.add_argument('--model', type=str, default='model_20h.json',
                    help='model to retrieve')
args = parser.parse_args()
"""
### RETRIEVE DATA
if len(sys.argv) == 1:
    path = 'MNIST.mat'
else:
    path = sys.argv[1]
print('Loading {}'.format(path))
data = spio.loadmat(path)
X, y = data['input_images'], data['output_labels']

### SET DEVICE
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: {}'.format(device))

### RETRIEVE MODEL
params = torch.load('pre_trained_model//model_20h.json')
print('Number of hidden units: {}'.format(params['net_params']['z_dim']))
model = VAE.VAE(**params['net_params']).to(device)
model.load_state_dict(params['net_state'])
model.eval()

### COMPUTE MSELOSS
loss_fn = torch.nn.MSELoss()
X = torch.tensor(X)
X = Variable(X.view(X.size(0), -1)).to(device)
out, _, _ = model(X.float())
loss = loss_fn(out, X)
print('MSELoss (pytorch function): {}'.format(np.round(loss.detach().cpu().numpy(), 3)))

### GENERATE RECON DATA IN .MAT
print('Generating recon_MNIST.mat file...')
X_out = out.detach().cpu().numpy()
mnist_dict = {'input_images': X_out, 'output_labels': y}
spio.savemat('recon_MNIST.mat', mnist_dict)
print('Done')
