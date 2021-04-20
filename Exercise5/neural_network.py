# Use Python 3.8 or newer (https://www.python.org/downloads/)
from typing import List
import unittest
import numpy as np
import pickle
import os
from math import e


class Node:
    """ Class representing a node in the NeuralNetwork."""

    def __init__(self, N: int) -> None:
        """ Initialize a node.

        :param N: Number of weights from this node to the next layer.
        :return: None.
        """
        self.N = N
        self.a = 0  # activation value

        if self.N:  # a node can have no weights to next layer if it is in output layer
            self.weights = np.zeros(self.N)  # initialize weights

    def randomize_weights(self):
        """ Randomizes the weights of this node between -0.5 and 0.5 """

        self.weights = np.random.rand(self.N)-0.5


def g(x: float) -> float:
    """ Sigmoid activation function.

    :return: Sigmoid function value at x.
    """
    return 1/(1+e**(-x))


def dg(x: float) -> float:
    """ Derivative of sigmoid activation function.

    :return: Sigmoid derivative function value at x.
    """
    return g(x)*(1-g(x))


class NeuralNetwork:

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.

        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-2  # increased a bit to achieve higher accuracy.

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer

        # List because of modularity (if there should be more output nodes)
        self.output_nodes = [Node(0)]

        # Initialize the layers with nodes depending on hidden layer or not
        if self.hidden_layer:
            self.hidden_nodes: List[Node] = [
                Node(1) for _ in range(self.hidden_units)]

            self.input_nodes: List[Node] = [
                Node(self.hidden_units) for _ in range(self.input_dim+1)]  # +1 for bias node

            self.layers = [self.input_nodes,
                           self.hidden_nodes, self.output_nodes]
        else:
            self.input_nodes: List[Node] = [
                Node(1) for _ in range(self.input_dim+1)]  # +1 for bias node
            self.layers = [self.input_nodes, self.output_nodes]

        self.L = len(self.layers)
        self.input_nodes[-1].a = 1  # Set bias activation to always be 1

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def propagate_forward(self, x: np.ndarray) -> List[List[float]]:
        """ Propagate x forward in the network.

        :param x: Input to propagate forward.
        :return: Weighted sum for each node.
        """

        weighted_sum: List[List[float]] = [[] for _ in range(
            self.L)]  # list of weighted sum for each node
        # note that the first list will always be empty since this is the input layer
        # it is only there to make indexing more intuitive

        # propagate input into input layer
        for i in range(self.input_dim):
            self.input_nodes[i].a = x[i]

        # for each other node in hidden and output layer calculate the activation
        for i in range(1, self.L):
            for j, node in enumerate(self.layers[i]):
                # calculate weighted sum
                in_j = sum([k.a*k.weights[j] for k in self.layers[i-1]])
                weighted_sum[i].append(in_j)
                node.a = g(in_j)  # calculate activation for this node

        return weighted_sum

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network.
        The runtime could be improved using NumPy arrays more effectively.
        On a decent CPU the runtime should still be pretty low, ~150s @ 2.3 GHz Intel i5.
        It could have been more optimized using more numpy arrays.
        """

        # randomize the weights for each node (between -0.5 and 0.5)
        for l in self.layers:
            for node in l:
                node.randomize_weights()

        # repeat learning step for epochs time
        for _ in range(self.epochs):

            # go through training data
            for x, y in zip(self.x_train, self.y_train):
                # propagate forward to find activations and weighted sums
                weighted_sums = self.propagate_forward(x)

                # initialize error for each node
                deltas = [[] for _ in range(self.L)]

                # propagate backward to output layer
                for j, node in enumerate(self.output_nodes):
                    deltas[self.L -
                           1].append(dg(weighted_sums[self.L-1][j])*(y-node.a))
                    # ∆[j]←g′(inj) × (yj − aj)

                # propagate backward through rest of layers
                for l in range(self.L-2, 0, -1):
                    for i, node in enumerate(self.layers[l]):
                        deltas[l].append(dg(weighted_sums[l][i])*sum(
                            [node.weights[j]*deltas[l+1][j] for j in range(len(self.layers[l+1]))]))
                        # ∆[i] ← g′(ini) sum_j (wi,j ∆[j])

                # update weights
                for l in range(self.L-1):
                    for node in self.layers[l]:
                        for k in range(node.N):
                            node.weights[k] += self.lr*node.a*deltas[l+1][k]
                            # wi,j←wi,j + α × ai × ∆[j]

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.

        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        self.propagate_forward(x)  # propagate forward
        return self.output_nodes[0].a  # return activation of output node


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        print("Accuracy perceptron: ", accuracy)  # TODO: Remove
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        print("Accuracy hidden: ", accuracy)  # TODO: Remove
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
