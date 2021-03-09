from re import sub
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import copy
import math

from graphviz import Digraph

continuous_variables = ['Age', 'Parch', 'SibSp']

def B(q):
    if q == 1.0 or q == 0.0: # No uncertainty check
        return 0 # No uncertainty has entropy 0
    return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def gain(A, examples):
    if A in continuous_variables:
        highest = 0
        v_k_max = 0
        for v_k in examples[A].unique(): # Go through all possible splits

            exs = copy.deepcopy(examples)
            exs[A] = (exs[A] >= v_k).astype(int)
            p = exs['Survived'].value_counts()[1]
            n_p_sum = exs['Survived'].shape[0]
            gain =  B(p/n_p_sum) - remainder(A, exs, n_p_sum)
            if gain > highest:
                highest = gain
                v_k_max = v_k

        return highest, v_k_max

    else:
        p = examples['Survived'].value_counts()[1]
        n_p_sum = examples['Survived'].shape[0]
        return B(p/n_p_sum) - remainder(A, examples, n_p_sum), NaN

def remainder(A, examples, n_p_sum):
    k = examples.groupby([A])['Survived'].value_counts()
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
    if examples['Survived'].nunique() == 1:
        return True, examples['Survived'].unique()[0]
    else:
        return False, NaN

class DT():

    def __init__(self, name):
        self.name = name
        self.value = NaN
        self.labels = {}

    def classify(self, data):
        tree = self
        while True:
            if not math.isnan(tree.value): # continuous variable
                if data[tree.name] >= tree.value:
                    k = tree.labels[">=" + str(tree.value)]
                else:
                    k = tree.labels["<" + str(tree.value)]
            else: # discrete variable
                k = tree.labels[data[tree.name]] # fix key error (Parch/SibSp)
            if isinstance(k, type(DT)):
                tree = k
            else:
                return k

    @staticmethod
    def graph_tree(tree, dot):

        for key, val in tree.labels.items():
            dot.node(str(tree.name)+str(counter[tree.name]), str(tree.name))

            if val == 1:
                dot.node(str(val)+str(counter[val]), 'Survived')
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
}

training_data = pd.read_csv('Exercise4/train.csv')

#print(training_data['Age'].median())
#print(training_data['Survived'].mean())
#print(training_data.groupby(['Pclass'])['Sex'].value_counts())

def plurality_value(examples):
    k = examples['Survived'].value_counts()
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

    truth_val, classification = check_same_classification(examples)
    if not examples.shape[0]:
        return plurality_value(parent_examples)
    elif truth_val:
        return classification
    elif not attributes:
        return plurality_value(examples)
    else:
        gains = [gain(a, examples) for a in attributes]
        max_A = max(gains, key=lambda item:item[0])
        A_index = gains.index(max_A)
        A = attributes[A_index]
        tree = DT(A)
        new_attr = [x for x in attributes if x != A]
        #print("Gains: ", gains)
        #print("Attributes: ", attributes)
        if not math.isnan(max_A[1]):
            tree.value = max_A[1]
            new_attr = [x for x in attributes if x != A]
            exs = examples[examples[A] >= max_A[1]]
            new_attr_copy = [x for x in new_attr]
            subtree = DTL(exs, new_attr_copy, examples)
            tree.labels[">=" + str(max_A[1])] = subtree

            exs = examples[examples[A] < max_A[1]]
            new_attr_copy = [x for x in new_attr]
            subtree = DTL(exs, new_attr_copy, examples)
            tree.labels["<" + str(max_A[1])] = subtree
        else:
            for v_k in training_data[A].unique():
                exs = examples[examples[A] == v_k]
                new_attr_copy = [x for x in new_attr]
                subtree = DTL(exs, new_attr_copy, examples)
                tree.labels[v_k] = subtree

        return tree

def test(DT, test_data):
    correct = 0
    for i in range(test_data.shape[0]):
        DT.classify(test_data.iloc[i])
        if DT.classify(test_data.iloc[i]) == test_data.iloc[i]['Survived']:
            correct += 1
    print(f'Accuracy: {round(100*correct/test_data.shape[0],1)}%')


if __name__=='__main__':

    training_data = pd.read_csv('Exercise4/train.csv')

    training_data = training_data.dropna() # drop all rows with na data
    attributes = list(training_data.columns.values)
    attributes.remove('Survived') # Output attribute

    # Input attributes:

    # Continuous attributes
    attributes.remove('Name')
    #attributes.remove('Age')
    attributes.remove('Cabin')
    attributes.remove('Ticket')
    attributes.remove('Fare')

    # Discrete attributes
    #attributes.remove('SibSp')
    #attributes.remove('Parch')
    attributes.remove("Embarked")
    attributes.remove("Pclass")
    #attributes.remove("Sex")

    # Testing
    #training_data['Age'] = (training_data['Age'] > 70).astype(int)
    #training_data['Parch'] = (training_data['Parch'] > 1).astype(int)
    #training_data['SibSp'] = (training_data['SibSp'] > 1).astype(int)
    #print(training_data[training_data['Sex']=='male'].groupby(['Age'])['Survived'].value_counts())



    test_data = pd.read_csv('Exercise4/test.csv')
    #print(test_data[test_data['Sex']=='male'].groupby(['Age'])['Survived'].value_counts())

    dot = Digraph(comment='Network')
    DT = DTL(training_data, attributes, [])
    DT.graph_tree(DT, dot)
    dot.save('test.gv')
    dot.render('test.gv', view=True)

    test(DT, test_data)

