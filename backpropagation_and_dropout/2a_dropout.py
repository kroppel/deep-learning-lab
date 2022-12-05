# -*- coding: utf-8 -*-
"""# 2a - Dropout
In this exercise you will modify the structure seen previously to add the Dropout Functionality

## Comment references:

- [1] Dropout paper:
  https://proceedings.mlr.press/v48/gal16.html
"""

import random
import math
import torch
import numpy as np
import matplotlib

activation_function = "sigmoid"



def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt

class Neuron:
    def __init__(self, bias, drop_prob=0.01):
        self.bias = bias
        self.weights = []
        self.drop_prob = drop_prob

    def calculate_output(self, inputs, train=False):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input(train=train))
        if train:
            self.output = self.dropout(self.output, self.drop_prob)
        return self.output

    def dropout(self, output, drop_probability): 
        return output * (random.random() >= drop_probability)

    def calculate_total_net_input(self, train=False):
        total = 0
        for i in range(len(self.inputs)):
            if train:
                total += self.inputs[i] * self.weights[i]
            else:
                total += self.inputs[i] * self.weights[i] * (1-self.drop_prob)
        return total + self.bias

    # Apply the activation function to the output of the neuron
    def squash(self, total_net_input):
        if activation_function == "relu":
            # RELU
            return max(0, total_net_input)
        elif activation_function == "sigmoid":
            # LOGISTIC
            return 1 / (1 + math.exp(-total_net_input))
        elif activation_function == "tanh":
            return tanh(total_net_input)[0]
        elif activation_function == "relu_clipped":
            return np.clip(max(0, total_net_input),0,1)
        raise NotImplementedError

    # Determine how much the neuron's total input has to change to move closer to the expected output
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return (
            self.calculate_pd_error_wrt_output(target_output)
            * self.calculate_pd_output_wrt_total_net_input()
        )

    # The error for each neuron is calculated by a version of the Mean Square Error method:
    def calculate_error(self, target_output):
        MSE = 0.5 * (target_output - self.output) ** 2
        return MSE

    # The partial derivate of the error with respect to actual output then is calculated by:
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    def calculate_pd_output_wrt_total_net_input(self):
        if activation_function == "relu" :
            # RELU derivative
            return 1 if self.output > 0 else 0
        elif activation_function == "sigmoid":
            # LOGISTIC derivative
            return self.output * (1 - self.output)
        elif activation_function == "tanh":
            return tanh(self.output)[1]
        elif activation_function == "relu_clipped" :
            # RELU derivative
            return 1 if self.output > 0 and self.output < 1 else 0
        else:
            raise NotImplementedError()

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

###


class NeuronLayer:
    def __init__(self, num_neurons, bias, drop_prob=0.01):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias, drop_prob=drop_prob))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs, train=False):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs, train=train))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs


class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None, drop_prob=0.01):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias, drop_prob=drop_prob)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias, drop_prob=0)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs, train=False):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs, train=train)
        return self.output_layer.feed_forward(hidden_layer_outputs, train=train)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs, train=True)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_output_wrt_total_net_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets, train=False):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs, train=train)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error


# Train

from matplotlib import pyplot as plt
import pandas as pd


drop_prob = 0
in_size = 2
hidden_size = 2
out_size = 2

hidden_weights = list(np.random.standard_normal(in_size*hidden_size))
output_weights = list(np.random.standard_normal(hidden_size*out_size))

activation_function = "relu_clipped"
nn_nodrop = NeuralNetwork(
    in_size, hidden_size, out_size,
    hidden_layer_bias=0.01,
    hidden_layer_weights= hidden_weights.copy(),
    output_layer_weights= output_weights.copy(),
    output_layer_bias=0.99,
    drop_prob=drop_prob
)
nn_nodrop.LEARNING_RATE = 0.05

print(f"Learning with a FullyConnected NN with activation={activation_function}, and dropout prob {drop_prob}")

fig, ax = plt.subplots()

losses_nodrop = []
epochs = 3000
for i in range(epochs):
    nn_nodrop.train([0.05, 0.1], [0.01, 0.99])
    loss = nn_nodrop.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]], train=False)
    losses_nodrop.append(loss)
    if i % 300 == 0:
        print("Epoch:", i, " Loss: ", loss)
    if round(loss, 9) == 0:
        print("Epoch:", i, " Loss: ", loss)
        print("Reached convergence!")
        break


colors = matplotlib.cm.get_cmap("tab10")



drop_prob = 0.1

#activation_function = "tanh"
nn_withdrop = NeuralNetwork(
    in_size, hidden_size, out_size,
    hidden_layer_bias=0.01,
    hidden_layer_weights= hidden_weights.copy(),
    output_layer_weights= output_weights.copy(),
    output_layer_bias=0.99,
    drop_prob=drop_prob
)
nn_withdrop.LEARNING_RATE = 0.05

print(f"Learning with a FullyConnected NN with activation={activation_function}, and dropout prob {drop_prob}")

losses_drop = []
epochs = 3000
for i in range(epochs):
    nn_withdrop.train([0.05, 0.1], [0.01, 0.99])
    loss = nn_withdrop.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]], train=False)
    losses_drop.append(loss)
    if i % 300 == 0:
        print("Epoch:", i, " Loss: ", loss)
    if round(loss, 9) == 0:
        print("Epoch:", i, " Loss: ", loss)
        print("Reached convergence!")
        break
    
ax.plot(range(len(losses_nodrop)), losses_nodrop, label='No Dropout', color=colors(0))
ax.plot(range(len(losses_drop)), losses_drop, label='Dropout', color=colors(1))
plt.legend(fontsize=16)
