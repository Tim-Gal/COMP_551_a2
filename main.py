import matplotlib
import numpy as np

import torch
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
#from tqdm import notebook
#from jedi.api.refactoring import inline
#%matplotlib notebook
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
])


mnist_trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=mnist_transform)

cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=mnist_transform)
cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=mnist_transform)


batch_size = 64
mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

cifar_train_loader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=False)

print(type(mnist_trainset))
print(type(mnist_test_loader))

class Perceptron:

    def __init__(self, add_bias=True, max_iters=10000, record_updates=False):
        self.max_iters = max_iters
        self.add_bias = add_bias
        self.record_updates = record_updates
        if record_updates:
            self.w_hist = []  # records the weight
            self.n_hist = []  # records the data-point selected

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x, np.ones(N)])
        N, D = x.shape
        w = np.zeros(D)  # initialize the weights
        if self.record_updates:
            w_hist = [w]
            # y = np.sign(y -.1)                             #to get +1 for class 1 and -1 for class 0
        y = 2 * y - 1  # converting 0,1 to -1,+1
        t = 0
        change = True  # if the weight does not change the algorithm has converged
        while change and t < self.max_iters:
            change = False
            for n in np.random.permutation(N):
                yh = np.sign(np.dot(x[n, :], w))  # predict the output of the training sample
                if yh == y[n]:
                    continue  # skip the samples which are correctly classified
                # w = w + (y[n]-yh)*x[n,:]               #update the weights
                w = w + y[n] * x[n, :]
                if self.record_updates:
                    self.w_hist.append(w)
                    self.n_hist.append(n)
                change = True
                t += 1
                if t >= self.max_iters:
                    break
        if change:
            print(f'did not converge after {t} updates')
        else:
            print(f'converged after {t} iterations!')
        self.w = w
        return self

    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x, np.ones(Nt)])
        yh = np.sign(np.dot(self.w, x))
        return (yh + 1) // 2  # converting -/+1 to classes 0,1