from re import sub
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np

from graphviz import Digraph

def B(q):
    if q == 1.0 or q == 0.0: # No uncertainty check
        return 0
    return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def gain(A, examples):
    p = examples['WillWait'].value_counts()[1]
    n_p_sum = examples['WillWait'].size
    return B(p/n_p_sum) - remainder(A, examples, n_p_sum)

def remainder(A, examples, n_p_sum):
    k = examples.groupby([A])['WillWait'].value_counts()
    s = 0
    for i in examples[A].unique():
        try:
            p_i = k[i][1]
        except:
            p_i = 0

        try:
            n_i = k[i][0]
        except:
            n_i = 0

        sum_ni_pi = p_i + n_i
        s += sum_ni_pi * B(p_i/sum_ni_pi)
    return s/n_p_sum

def check_same_classification(examples):
    if examples['WillWait'].nunique() == 1:
        return True, examples['WillWait'].unique()[0]
    else:
        return False, NaN

class DT():

    def __init__(self, name):
        self.name = name
        self.labels = {}

    def classify(self, data):
        tree = self
        while True:
            try: # TODO?
                k = tree.labels[data[tree.name]]
            except:
                return 0
            if isinstance(k, type(DT)):
                tree = k
            else:
                return k

    @staticmethod
    def graph_tree(tree, dot):

        for key, val in tree.labels.items():
            dot.node(str(tree.name)+str(counter[tree.name]), str(tree.name))
            if val == 1:
                dot.node(str(val)+str(counter[val]), 'Yes')
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val)+str(counter[val]), str(key))
            elif val == 0:
                dot.node(str(val)+str(counter[val]), 'No')
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val)+str(counter[val]), str(key))
            else:
                dot.node(str(val.name)+str(counter[val.name]), str(val.name))
                dot.edge(str(tree.name)+str(counter[tree.name]), str(val.name)+str(counter[val.name]), str(key))
                DT.graph_tree(val, dot)
                counter[val.name] += 1

        counter[1] += 1
        counter[0] += 1





counter = {
    0: 0,
    1: 0,
    "Patrons": 0,
    "Hungry": 0,
    "Type": 0,
    "Fri": 0,
    "Age": 0,
    "Name": 0,
    "Cabin": 0,
    "Ticket": 0,
    "Fare": 0,
    "Pclass": 0,
    "Parch": 0
}

training_data = pd.read_csv('Exercise4/train.csv')

#print(training_data.groupby(['Pclass'])['Sex'].value_counts())

def DTL(examples, attributes, parent_examples):

    truth_val, classification = check_same_classification(examples)
    if not examples.shape[0]:
        k = parent_examples.WillWait.value_counts()

        try:
            p_i = k[1]
        except:
            p_i = 0

        try:
            n_i = k[0]
        except:
            n_i = 0
        if p_i >= n_i:
            return 1
        else:
            return 0

    elif truth_val:
        return classification

    elif not attributes:
        k = examples.WillWait.value_counts()

        try:
            p_i = k[1]
        except:
            p_i = 0

        try:
            n_i = k[0]
        except:
            n_i = 0

        if p_i >= n_i:
            return 1
        else:
            return 0
    else:
        A_index = np.argmax([gain(a, examples) for a in attributes])
        A = attributes[A_index]
        tree = DT(A)

        #print("Most gains: ", A)

        new_attr = [x for x in attributes if x != A]

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
        if DT.classify(test_data.iloc[i]) == test_data.iloc[i]['WillWait']:
            correct += 1
    print(correct)
    print(correct/test_data.shape[0])


if __name__=='__main__':

    training_data = pd.read_csv('Exercise4/book.csv')

    training_data = training_data.dropna() # drop all rows with na data
    attributes = list(training_data.columns.values)
    attributes.remove('WillWait')

    test_data = pd.read_csv('Exercise4/book.csv')

    dot = Digraph(comment='Network')
    DT = DTL(training_data, attributes, [])
    DT.graph_tree(DT, dot)
    dot.save('test.gv')
    dot.render('test.gv', view=True)

