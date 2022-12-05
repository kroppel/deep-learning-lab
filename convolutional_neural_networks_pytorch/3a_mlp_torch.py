"""
# MLP Torch implementation
In this notebook you will see how to implement the neural network from previous lessons using pytorch.
You will also see how a model is usually trained

## Dataset
The first thing we need to do is understand what data we are trying to learn on.

For this exercise, a small version of MNIST dataset is given (`notMNIST_small.npy`). 
This dataset contains images of different character. Each character is assigned a class from 0 to 9, and every class has multiple samples.

We will now load the dataset and prepare it for the operations. 
For this purpose, the `utils.py` file contains a DataGenerator, 
which will take care of loading the dataset and splitting it into train and test splits.
"""

# imports
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from torch import nn

# Step 1: Define the dataset
from utils import DataGenerator
from torch.utils.data import TensorDataset, DataLoader
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
x_train_img = x_train.reshape(x_train.shape[0], 1, size, size)
x_test_img = x_test.reshape(x_test.shape[0], 1, size, size)

# plot an image
fig, axs = plt.subplots(nrows=2, ncols=5)
for i in range(10):
    axs[i%2, i//2].set_title(f'Label: {y_test[i]}')
    axs[i%2, i//2].imshow(x_test_img[i, :, :].squeeze(), cmap='gray', vmin=0, vmax=255)

plt.show()

# znorm
mu = np.mean(x_train)
std = np.std(x_train)
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

# transform the data in a Tensor (for torch operations)
x_train_tens = torch.from_numpy(x_train)
y_train_tens = torch.from_numpy(y_train)

x_test_tens = torch.from_numpy(x_test)
y_test_tens = torch.from_numpy(y_test)

print(x_train_tens.shape)

"""
## Model
The next piece needed is the model itself.
In pytorch, each model inherits from the class Module. This means that every model should implement 2 functions: `__init__` and `forward`.

- `__init__` is the constructor of our model. It should take as inputs the hyperparameters of the model and will build the layers accordingly.
- `forward` is the forward pass of the model. It takes as input the data (batched) and returns the outputs of the output layer.

You will note that we are not going to define the backpropagation in our model. 
That is because the gradient calculation and backpropagation algorithms are handled by pytorch functions that are external and independent to the model.

We are going to implement a MLP, and to do so, we will use Fully Connected layers, called Linear layers in pytorch. 
Keep in mind that a Linear layer **is NOT** defining the number of neurons, but it's defining the weight matrix. 
In pytorch, there are no actual neurons.
"""

class MyMLP(nn.Module):
    
    def __init__(self, num_hidden_layers : int = 1, input_size : int = 2, hidden_size : int = 2, output_size : int = 2) -> None:
        """
        The constructor takes as inputs the parameters of the network and builds its layers.
        """
        # the super() function calls the constructor of nn.Module. It's necessary to initialize some torch parameters.
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        self.first_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.ReLU = nn.ReLU()

        # modulelist is a list of layers. You can execute each layer independently by just calling it (see forward)
        # it's handy for when you want to create a bulk of layers based on a hyperparameter (like now).
        # Sequential is also a list of layers. However all the layers inside a Sequential module will be executed one after the other
        self.hidden_layers = nn.ModuleList(
            [
            nn.Sequential(nn.Linear(in_features=hidden_size, out_features=hidden_size), 
            nn.ReLU()) 
            for _ in range(num_hidden_layers)]
        )

        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        """
        The forward function is the forward pass of the network. 
        This decides the actual flow of data through the network.
        """
        # calculate the output of the first layer
        first_layer_out = self.first_layer(x)
        # non-linearity
        first_layer_out = self.ReLU(first_layer_out)

        # feed the first layer output to the hidden layer(s)
        # first get output of first hidden layer
        # this is executing the first Sequential module inside the ModuleList
        hidden_layers_out = self.hidden_layers[0](first_layer_out)
        # then iterate over all the others
        for i in range(1, self.num_hidden_layers):
            hidden_layers_out = self.hidden_layers[i](hidden_layers_out)

        # get output
        output = self.output_layer(hidden_layers_out)

        # return the output
        return output


x = torch.Tensor([0.05, 0.1])
model = MyMLP(input_size=2, num_hidden_layers=1, hidden_size=2, output_size=2)
y = model(x)
print(y)


"""
## Train the model
We need to define a few more things and then we can start taining:

- `Dataset`: We need to define proper pytorch datasets (train and test). Luckily pytorch has a handy function for that: TensorDataset.
- `Dataloader`: We then need to define the train and test dataloaders. These object are responsible for iterating over the datasets and collecting the input data and GT.
- `Criterion`: This is the Loss function that we want to use for our task and to calculate the total error.
- `Optimizer`: This is the core of the training: it is responsible for the backpropagation and update of our weights! (No more manual backprop, YAY!) 
"""

#define the model
input_size = x_train_tens[0].numel()
num_classes = len(set(y_train))
model = MyMLP(input_size=input_size, num_hidden_layers=2, hidden_size=10, output_size=num_classes)


# Define training parameters: optimizer, loss, epochs, batch size
epochs = 30
# it's a classification problem, so cross entropy is good
loss_function = nn.CrossEntropyLoss()
# SGD is an optimizer based on Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# SGD operates on batches of data, meaning that it calculates the error per batch, and updates the weights each batch
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
