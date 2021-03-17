from re import sub
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import copy
import math

from graphviz import Digraph

continuous_variables = ['Age', 'Cabin', 'Fare', 'Ticket', 'Parch', 'SibSp'] # change this list to test between discrete and cont. variables

GOAL_ATTRIBUTE = 'Survived'

def B(q):
    """ Implements the function B(q) from AIMA """

    if q == 1.0 or q == 0.0: # No uncertainty check
        return 0 # No uncertainty has entropy 0
    return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def gain(A, examples):
    """ Implements the function Gain(A) from AIMA """

    if A in continuous_variables:
        highest = 0
        val_max = 0
        unique_vals = examples[A].unique()
        unique_vals.sort()

        for i in range(1, unique_vals.shape[0]): # Go through all possible splits
            val =(unique_vals[i] + unique_vals[i-1])/2
            exs = copy.deepcopy(examples)
            exs[A] = (exs[A] >= val).astype(int) # create split
            p = exs[GOAL_ATTRIBUTE].value_counts()[1]
            n_p_sum = exs[GOAL_ATTRIBUTE].shape[0]
            gain =  B(p/n_p_sum) - remainder(A, exs, n_p_sum)
            if gain > highest:
                highest = gain
                val_max = val

        return highest, val_max

    else:
        p = examples[GOAL_ATTRIBUTE].value_counts()[1]
        n_p_sum = examples[GOAL_ATTRIBUTE].shape[0]
        return B(p/n_p_sum) - remainder(A, examples, n_p_sum), NaN

def remainder(A, examples, n_p_sum):
    """ Implements the function Remainder(A) from book """

    k = examples.groupby([A])[GOAL_ATTRIBUTE].value_counts()
    s = 0
    for i in examples[A].unique():
        p_i = 0
        n_i = 0
        if 1 in k[i].keys():
            p_i = k[i][1]
        if 0 in k[i].keys():
            n_i = k[i][0]

        sum_ni_pi = p_i + n_i
        s += sum_ni_pi * B(p_i/sum_ni_pi)
    return s/n_p_sum

def check_same_classification(examples):
    """ Checks if all of the examples have the same classification and if so returns True and this classification, else returns False and NaN """

    if examples[GOAL_ATTRIBUTE].nunique() == 1:
        return True, examples[GOAL_ATTRIBUTE].unique()[0]
    else:
        return False, NaN

class DT():
    """ Decision tree class """

    def __init__(self, name):
        self.name = name # Name of this tree
        self.value = NaN # Value used if the tree is just 0 or 1 for Goal attribute
        self.labels = {} # Recursive dictionary which contains other DTs

    def classify(self, data):
        """ Method for getting the classification of an example """

        tree = self
        while True:
            if not math.isnan(tree.value): # continuous variable
                if data[tree.name] >= tree.value:
                    k = tree.labels[">=" + str(tree.value)]
                else:
                    k = tree.labels["<" + str(tree.value)]
            else: # discrete variable
                if data[tree.name] in tree.labels.keys():
                    k = tree.labels[data[tree.name]]
                else:
                    return 0
            if isinstance(k, type(DT)):
                tree = k
            else:
                return k

    @staticmethod
    def graph_tree(tree, dot):
        """ Taks in a DT and graphviz dot-object as a reference and draws the graph of the tree """

        for key, val in tree.labels.items():
            dot.node(str(tree.name)+str(counter[tree.name]), str(tree.name))

            if val == 1:
                dot.node(str(val)+str(counter[val]), GOAL_ATTRIBUTE)
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val)+str(counter[val]), str(key))
            elif val == 0:
                dot.node(str(val)+str(counter[val]), 'Died')
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val)+str(counter[val]), str(key))
            else:
                counter[1] += 1
                counter[0] += 1
                dot.node(str(val.name)+str(counter[val.name]), str(val.name))
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val.name)+str(counter[val.name]), str(key))
                DT.graph_tree(val, dot)
                counter[val.name] += 1

        counter[1] += 1
        counter[0] += 1

counter = {
    0: 0,
    1: 0,
    "Embarked": 0,
    "Sex": 0,
    "Parc": 0,
    "SibSp": 0,
    "Age": 0,
    "Name": 0,
    "Cabin": 0,
    "Ticket": 0,
    "Fare": 0,
    "Pclass": 0,
    "Parch": 0
} # Used for drawing graph with multiple nodes with the same name


def plurality_value(examples):
    """ Calculates plurality value of the examples, this is then returned """

    k = examples[GOAL_ATTRIBUTE].value_counts()
    p_i = 0
    n_i = 0
    if 1 in k.keys():
        p_i = k[1]
    if 0 in k.keys():
        n_i = k[0]

    if p_i >= n_i:
        return 1
    return 0



def DTL(examples, attributes, parent_examples):
    """ DTL algorithm from AIMA (Figure 18.5) """

    truth_val, classification = check_same_classification(examples)
    if not examples.shape[0]:
        return plurality_value(parent_examples)
    elif truth_val:
        return classification
    elif not attributes:
        return plurality_value(examples)
    else:
        # calculate the gains of each attribute (the continuous variables are handled inside gain)
        gains = [gain(a, examples) for a in attributes]
        max_A = max(gains, key=lambda item:item[0]) # find max gain
        A_index = gains.index(max_A)
        A = attributes[A_index]
        tree = DT(A) # new tree with this attribute
        new_attr = [x for x in attributes if x != A] # remove the new attribute


        if not math.isnan(max_A[1]): # continuous attribute was chosen
            tree.value = max_A[1]
            new_attr = [x for x in attributes if x != A]
            exs = copy.deepcopy(examples)

            # Subtree for examples >= the chosen value
            exs = exs[exs[A] >= max_A[1]]
            new_attr_copy = [x for x in new_attr]
            subtree = DTL(exs, new_attr_copy, examples)
            tree.labels[">=" + str(max_A[1])] = subtree

            # Subtree for examples < the chosen value
            exs = copy.deepcopy(examples)
            exs = exs[exs[A] < max_A[1]]
            new_attr_copy = [x for x in new_attr]
            subtree = DTL(exs, new_attr_copy, examples)
            tree.labels["<" + str(max_A[1])] = subtree
        else: # discrete attribute was chosen
            for v_k in training_data[A].unique():
                exs = examples[examples[A] == v_k]
                new_attr_copy = [x for x in new_attr]
                subtree = DTL(exs, new_attr_copy, examples)
                tree.labels[v_k] = subtree

        return tree

def test(DT, test_data):
    """ Takes in a DT and data to test the tree and returns the accuracy of the tree """

    correct = 0
    for i in range(test_data.shape[0]):
        DT.classify(test_data.iloc[i])
        if DT.classify(test_data.iloc[i]) == test_data.iloc[i][GOAL_ATTRIBUTE]:
            correct += 1
    print(f'Accuracy: {round(100*correct/test_data.shape[0],1)}%')


if __name__=='__main__':

    training_data = pd.read_csv('Exercise4/train.csv')

    attributes = list(training_data.columns.values)
    attributes.remove(GOAL_ATTRIBUTE) # Output attribute

    # Input attributes: (comment out to add attribute, decomment to remove)

    # Continuous attributes
    attributes.remove('Name')
    attributes.remove('Age') # removed since missing values
    attributes.remove('Cabin') # removed since missing values
    attributes.remove('Ticket') # not good for splitting
    #attributes.remove('Fare')

    # Discrete attributes
    #attributes.remove('SibSp')
    #attributes.remove('Parch')
    attributes.remove("Embarked") # removed since irrelevant
    #attributes.remove("Pclass")
    #attributes.remove("Sex")

    test_data = pd.read_csv('Exercise4/test.csv')

    dot = Digraph(comment='Network')
    DT = DTL(training_data, attributes, []) # create DT
    DT.graph_tree(DT, dot)
    dot.save('test5.gv')
    dot.render('test5.gv', view=True)

    test(DT, test_data) # test DT on test dataset

