"""
# Convolutional Neural Networks
Convolutional Neural Networks make use of the Convolution operation to operate on the inputs.
Imagine that we want to classify an audio signal. 
To do that with Fully-Connected Networks we would need to specify an input neuron for each "sample" of the audio signal. For instance, if the signal is 10 seconds long, recorded at 2 Hz (2 samples per second) we would need 20 input neurons to be able to process it in its entirity.

With Convolutional Networks, instead, we are defining a smaller kernel that will
slide over the entire input. This greatly reduces the number of parameters of the network, while also making it a bit slower.

In this notebook we are going to classify images from (a subset of) MNIST
"""

# import
import torch
from torch import nn
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

"""
## Define our Convolutional Model
We are going to define a Convolutional Model with 2 layers of convolutional operations and 1 fully connected layer in the end for classification.
The goal of this network will be to classify images, so we need to use 2D Convolution to operate on both the dimension of the images.

### Exercise 1
Implement the forward pass of the network.
The data needs to go through both Conv layers and then be flattened to be processed by the final FC layer.
"""
class ConvolutionalNet(nn.Module):
    # Define the model constructor
    def __init__(self, k1_size : int = 3, k2_size: int = 3, num_classes : int = 2, in_channels : int = 3, hidden_channels : int = 3, input_shape = (10,10), stride : int = 1, padding : int = 0) -> None:
        """
        The __init__ function takes as input the parameters (our choice) that we want for the network.
        It then builds all the layers of our network with the chosen parameters. 
        """
        super().__init__()
        self.num_classes = num_classes
        self.k1_size = k1_size
        self.k2_size = k2_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ReLU = nn.ReLU()
        self.k1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=k1_size, stride=stride, padding=padding)
        self.k2 = nn.Conv2d(in_channels=hidden_channels, out_channels=1, kernel_size=k2_size, stride=stride, padding=padding)

        # we need to know how big to make the Linear Layer, since its size depends on the output of k2.
        # we create a dummy input to pass throug the net with the same size as the input we are going to use
        dummy_x = torch.rand((1, in_channels, *input_shape))
        dummy_y = self.k2(self.k1(dummy_x))

        # we need to linearize the output
        dummy_y_lin = dummy_y.flatten()
        self.linear = nn.Linear(dummy_y_lin.shape[0], num_classes)
    
    # define the forward function
    def forward(self, x):
        """
        The forward function takes the input and passes it through the layers.
        Pytorch takes care of calculating the gradients.
        TODO: implement forward pass
        """
        out_k1 = self.ReLU(self.k1(x))
        out_k2 = self.ReLU(self.k2(out_k1))
        
        # we need to linearize the input
        out_k2_lin = out_k2.flatten(start_dim=1)
        out_class = self.linear(out_k2_lin)
        return out_class

### Test that our network works
input_shape = (20,20)
in_channels = 3
model = ConvolutionalNet(k1_size=3, k2_size=3, num_classes=2, in_channels=in_channels, hidden_channels=2, input_shape=input_shape, stride=1, padding=0)
x = torch.rand((1, in_channels, *input_shape))
out = model(x)
print(out)


"""
## Training a Convolutional Network
Convolutional Networks are trained the same way as MLPs.
1. Take a Dataset (collection of samples)
1. Split the Dataset into a _train and _test partition
1. Separate the Indipendet variables (X) from the Dipendent variable(s) (Y). At this point you should have X_train, Y_train, X_test, Y_test
1. Iterate over each sample of the training split. At each step, compute the Loss and do Backpropagation
1. Once you have iterated over the whole train_split, do the same with the test_split, **without computing gradients** and **without doing Backpropagation**.
1. Repeat step 4 and 5 until you are happy of the results (mind the overfitting!).

In step 6, we are using the concept of Epoch: each iteration over the entire training set is called an epoch.
For most application, a single epoch is not sufficient for the network to learn, so we use many epochs. 
How many? This does not have an exact answer: ideally, you want to avoid overfitting (meaning that the model perfectly learns the training set, but loses the ability to generalize). This can be spotted when, during testing (step 5), the loss starts to continuosly rise with each epoch. This means that you should have stopped before then.
"""

# Step 1: Define the dataset
from utils import DataGenerator
from torch.utils.data import TensorDataset, DataLoader
# from tqdm import tqdm
from sklearn.metrics import accuracy_score
dg = DataGenerator()
dataset = dg.load("notmnist")

# Step 2: split into train and test splits
trainset, testset = dataset

# Step 3: separate X and Y
# prepare train data
x_train = trainset[:,:-1]
y_train = trainset[:,-1].reshape(-1)
# prepare test data
x_test = testset[:,:-1]
y_test = testset[:,-1].reshape(-1)

# extract data shape
nsamples = x_train.shape[0]
nfeatures = x_train.shape[1]
num_classes = len(set(y_train))

# reshape data to image format
size = int(np.sqrt(nfeatures))
x_train = x_train.reshape(x_train.shape[0], 1, size, size)
x_test = x_test.reshape(x_test.shape[0], 1, size, size)

# plot an image
fig, axs = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    axs[i%2, i//2].set_title(f'Label: {y_test[i]}')
    axs[i%2, i//2].imshow(x_test[i, :, :].squeeze(), cmap='gray', vmin=0, vmax=255)
plt.show()

# znorm
mu = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = x_train - mu
x_train = x_train / std
x_test = x_test - mu
x_test = x_test / std

# subsampling
nsamples = int(nsamples*0.1)
x_train = x_train[:nsamples]
y_train = y_train[:nsamples]
x_test = x_test[:nsamples]
y_test = y_test[:nsamples]

# check the data shape
num_channels = x_train.shape[1]
input_shape = (x_train.shape[2], x_train.shape[3])

# transform the data in a Tensor (for torch operations)
x_train_tens = torch.from_numpy(x_train)
y_train_tens = torch.from_numpy(y_train)

x_test_tens = torch.from_numpy(x_test)
y_test_tens = torch.from_numpy(y_test)


"""
## Train the model
This is the training procedure. This is a "general recipe" that you will most likely use whenever you will need to train a network, so look carefully and make an effort to understand every step!

### Exercise 2: fill put training procedure
- Define the network parameters
- Define the loss_function
- Implement the backward pass (using torch functions)
"""

# define the model
k1_size = 3
k2_size = 3
hidden_channels = 3
stride = 1
padding = 0

model = ConvolutionalNet(k1_size=k1_size, k2_size=k2_size, num_classes=num_classes,
            in_channels=num_channels, hidden_channels=hidden_channels, 
            stride=stride, padding=padding, input_shape=input_shape)

# Define training parameters: optimizer, loss, epochs, batch size
epochs = 15
# it's a classification problem, so cross entropy is good
loss_function = nn.CrossEntropyLoss()

# Adam is another popular optimizer that operates on batches
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 50

# Define a Dataset and a Dataloader
train_dataset = TensorDataset(x_train_tens, y_train_tens) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # create your dataloader
test_dataset = TensorDataset(x_test_tens, y_test_tens) # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # create your dataloader


# Step 4: iterate over the entire train_split, do backpropagation
train_losses = []
test_losses = []
train_accs = []
test_accs = []
for e in range(epochs):
    train_preds = np.array([])
    train_gts = np.array([])

    # set the model in train mode
    model.train()
    # each epoch iterates over the entire train spit
    for sample, target in train_dataloader:
        xin = sample.type(torch.FloatTensor)
        yin = target.type(torch.LongTensor)
        # forward pass - compute train predictions
        yout = model(xin)
        # compute loss value
        train_loss = loss_function(yout, yin)
        # backward pass - update the weights
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()        

        # save results for statistics
        preds = yout.argmax(axis=1)
        train_preds = np.concatenate((train_preds, preds.flatten().numpy()))
        train_gts = np.concatenate((train_gts, yin.flatten().numpy()))
    
    # Step 5: iterate over entire test_split and calculate test_loss
    test_preds = np.array([])
    test_gts = np.array([])

    # set model to eval mode (no backpropagation and gradient calculation, different behaviour of dropout etc.)
    model.eval()
    for sample, target in test_dataloader:
        xin = sample.type(torch.FloatTensor)
        yin = target.type(torch.LongTensor)
        # forward pass - compute train predictions
        yout = model(xin)
        # compute loss value
        test_loss = loss_function(yout, yin)

        # save results for statistics
        preds = yout.argmax(axis=1)
        test_preds = np.concatenate((test_preds, preds.flatten().numpy()))
        test_gts = np.concatenate((test_gts, yin.flatten().numpy()))
    
    # compute statistics every N epochs
    if e % 1 == 0:
        train_accs.append(accuracy_score(train_gts, train_preds))
        test_accs.append(accuracy_score(test_gts, test_preds))
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())


# Plot losses
fig, ax = plt.subplots()
colors = matplotlib.cm.get_cmap("tab10")   
ax.plot(range(len(train_losses)), train_losses, label='Train Loss', color=colors(0))
ax.plot(range(len(test_losses)), test_losses, label='Test Loss', color=colors(1))
plt.legend(fontsize=16)
plt.show()

# Plot Acc
fig, ax = plt.subplots()
colors = matplotlib.cm.get_cmap("tab10")   
ax.plot(range(len(train_accs)), train_accs, label='Train Acc', color=colors(0))
ax.plot(range(len(test_accs)), test_accs, label='Test Acc', color=colors(1))
plt.legend(fontsize=16)
plt.show()
