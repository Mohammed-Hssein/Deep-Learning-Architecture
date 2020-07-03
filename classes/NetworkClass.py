"""
This file entitled "NetworkClass.py" is basic implementation of a feed forward artificial
neural network.

@author: mohammed Hssein
"""

### Libraries

import random
import numpy as np

# Class Network

class Network(object):
    
    def __init__(self, network_size):
        """Docstring :
            
            __init__(self, network_size)
            
            Makes the initialization of an artificial neural network.
            
            Parameters
            ----------
            
            network_size : a pyhton list.
                    It contains a number of neurons in each layer, such that len(network_size)
                    is the number of layers on the current ANN architechture.
            
                    Precisely the k`th element of the list is the number of 
                    neurons in the k'the layer, therfore, the lenth of the list is 
                    the number of layers in the network architecture.
            
            Example
            -------
            >>> network = Network(network_size = [32, 16, 2])
                
                # This defines a network of 3 layers, with 32 neurons in
                # first layer, 16 in the second layer, and finally 2 in the 
                # output layer."""
        
        self.number_layers = len(network_size)
        self.network_size = network_size
        self.biases = [np.random.randn(y, 1) for y in network_size[1:]]
        self.weights = [np.random.randn(y, x)  
                               for x, y in zip(network_size[:-1], network_size[1:])]
    
    # Class method to define the network
    @classmethod
    def NetworkInitializer(self,networks_size_layers):
        """Docstring :
            
            NetworkInitializer(self,networks_size_layers)
            
            Makes the initialization of an artificial neural network.
            
            Parameters
            ----------
            
            network_size_layers : a pyhton list.
                    It contains a number of neurons in each layer, such that len(network_size_layers)
                    is the number of layers on the current ANN architechture.
            
                    Precisely the k`th element of the list is the number of 
                    neurons in the k'the layer, therfore, the lenth of the list is 
                    the number of layers in the network architecture.
            
            Example
            -------
            >>> import Network as net
            >>> network = net.NetworkInitializer(network_size_layers = [32, 16, 2])
                
                # This defines a network of 3 layers, with 32 neurons in
                # first layer, 16 in the second layer, and finally 2 in the 
                # output layer."""
                
        return Network(network_size=networks_size_layers)
    
    def  feedforward(self, input_data):
        """Docstring :
            feedforward(input_data)
            
            Returns the result of passing the input_data to the network.
            
            Parameters
            ----------
            input_data : 1-D array such that input_data.shape = network_size[0]
                    Represents the input data to the ANN architecture. Often, from
                    the training set or the test set.
            
            Returns
            -------
            output_data : 1-D array with same shape as input_data. 
                    Contains input_data transformed after going throw
                    the ANN architecture."""
        
        for b, w in zip(self.biases, self.weights):
            input_data = sigmoid(np.dot(w, input_data) + b)
        return input_data  #it's actually output data !
    
    
    def stochastic_gradien_descent(self, training_data, epochs, mini_batch_size,
                                     learning_rate, test_data = None):
        """Docstring :
            stochastic_gradien_descent(self, training_data, epochs, mini_batch_size,
                                     learning_rate, test_data = None)
            
            Launchs the training for the ANN architecture, with chosen params.
            
            Parameters
            ----------
            training_data : list type. Contains the training data to feed to the 
                    network architecture.
                    Example : training_data = [(2,6), (4,12), ... (entry, output)]
                    
            epochs : int type. Periods of training the networks. This means, the number times
                run the stochastique gradient descent algorithm to update the network
                weights.
                
            mini_batch_size : int type. Number of samples to feed to the network to change 
                    weights once. 
                    
            learning_rate : floate type. Between 0 and 1. The learning rate. How the learning 
                    is slow. Learning rate near 0 means the trainign will be long but, the weights 
                    are going to be well updated. a learning rate near 1 means the trainig will be 
                    faster, but the weights risks to not converge to desired optimal weights.
                    
            test_data : liste type (optional). Default None. If specified, networks establish 
                    comparaison between test_set values and values produces by the network
                    with current weights during training."""
                    
        if test_data : n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.__update_network_params(self, mini_batch, learning_rate)
            
            if test_data:
                print("Epoch {0}: {1}/{2} are correct".format(j, self.__evaluate(test_data), n_test))
            else : 
                print("Epoch {0} complete".format(j))
    
    
    def  __update_network_params(self, mini_batch, learning_rate):
        """Docstring : 
            __update_network_params(mini_batch, learning_rate)
            
            Updates the network weights during mini_batch feed, with the 
            learning_rate.
            
        Private method  : 
        -----------------
            
            Update the network's weights and biases using __backpropagation to a
            single mini batch.
            The ``mini_batch`` is a list of tuples ``(x, y)``, and ``learning_rate``
            is the learning rate.
            
            gradient = sum of gradients copputed for all arrors in mini-batchs
            """
            
        nabla_b = [np.zeros(b.shape()) for b in self.biases]
        nabla_w = [np.zeros(w.shape()) for w in self.weights]
        
        #Loop aiming to get nabla_b and nabla_w where gradients are computed
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.__backpropagation(self, x, y)
            #update nabla_b for each x in mini_batch of training
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            #update nabla_w for each x in mini_batch of training
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        #Update weights with w <-- w - [(learning_rate)/m]* gradient_w
        self.weights = [w - (learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        #Update biases with b <-- b - [(learning_rate)/m]* gradient_b
        self.biases = [b - (learning_rate/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def  __backpropagation(self, x, y):
        """Docstring : 
            __backpropagation(x, y)
            
            Backpropagate the error throw the network in order to update the weights.
            
            Two steps algorithm :
                First step is to perform the feedforward using the random
                (or actual) weights of the neuron.
                
                Second step is to perform the backpropagation of the error into the the
                network, so each weight updates itself according to it's contribution 
                to the error (dC/dw(l))
            Therfore, the training is performed, and by the end weights will be updated.
            
          Private method  :
          -----------------
                    Returns a tuple ``(nabla_b, nabla_w)`` representing the
                    gradient for the cost function C_x.  ``nabla_b`` and
                    ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
                    to ``self.biases`` and ``self.weights``.
                    
                    nabla_b = dC/db(l)
                    nabla_w = dC/dw(l)
                    
                    C is the cost function."""
                    
        nabla_b = [np.zeros(b.shape()) for b in self.biases] #full of dC/db(l)(j) set to 0
        nabla_w = [np.zeros(w.shape()) for w in self.weights] # full of dC/dw(l)(j,k) set to 0
        #feedforward
        activation = x
        activations = [x]
        z_vector = [] # for loop aims to compute a(l) and z(l) for the all layers
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_vector.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #__backpropagation of the error starting from the end (see equations BP i i=1,2,3,4)
        delta = self.__cost_derivative(activations[-1], y) * sigmoid_prime(z_vector[-1]) #BP 1
        nabla_b[-1] = delta # BP 3
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # BP 4
        
        for l in range(2, self.number_layers, 1):
            z = z_vector[-l]
            s_p = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*s_p # BP 2
            nabla_b[-l] = delta # BP3
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) # BP 4
        return nabla_b, nabla_w
    
    
    def __cost_derivative(self, output_activations, y):
        """DOcstring : 
            __cost_derivative(output_activations, y)
        
           Private method  :
           -----------------
                    FOR ONLY MSE LOSS FUNCTION
                    Return the vector of partial derivatives \partial C_x /
                    \partial a for the output activations."""
        return (output_activations-y)
    
    def __evaluate(self, test_data):
        """Docstring :
            
            __evaluate(self, test_data)
            
           Private method :
           ----------------
            Return the number of test inputs for which the neural
            network outputs the correct result. Note that the neural
            network's output is assumed to be the index of whichever
            neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
        
# Mathematical funcitons used 
def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function.
       We can easily prove this equality
       using basic derivation rules."""
    return sigmoid(z)*(1-sigmoid(z))

#----------------------------- END of Code --------------------------------------#
