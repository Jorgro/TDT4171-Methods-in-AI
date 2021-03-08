from numpy.core.numeric import NaN
import pandas as pd
import numpy as np

training_data = pd.read_csv('Exercise4/train.csv')

training_data = training_data.dropna() # drop all rows with na data


#training_data.value_counts()
#print(training_data['Survived'].nunique())
#print(training_data['Survived'].unique()[0])

#print(len(training_data['Survived']))
#print(training_data['Survived'].value_counts()[0])

""" print(training_data.groupby(['Embarked'])['Survived'].value_counts().iloc[0])
print(training_data.groupby(['Embarked'])['Survived'].value_counts().iloc[1])
print(training_data.groupby(['Embarked'])['Survived'].value_counts().iloc[2])
print(training_data.groupby(['Embarked'])['Survived'].value_counts().iloc[3])
print(training_data.groupby(['Embarked'])['Survived'].value_counts())
print(training_data['Embarked'].unique()) """

def B(q):
    if q == 1.0 or q == 0.0: # No uncertainty check
        return 0
    return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def gain(A, examples):
    p = examples['Survived'].value_counts()[1]
    n_p_sum = examples['Survived'].size
    return B(p/n_p_sum) - remainder(A, examples, n_p_sum)

def remainder(A, examples, n_p_sum):
    k = examples.groupby([A])['Survived'].value_counts()
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
    if examples['Survived'].nunique() == 1:
        return True, examples['Survived'].unique()[0]
    else:
        return False, NaN

class DT():

    def __init__(self, name):
        self.name = name
        self.labels = {}


def DTL(examples, attributes, parent_examples):

    truth_val, classification = check_same_classification(examples)
    if not examples.size:
        print("No examples")
        return parent_examples
    elif truth_val:
        print("Same classification")
        return classification
    elif not attributes:
        print("No attributes left")
        return examples
    else:
        A_index = np.argmax([gain(a, examples) for a in attributes])
        A = attributes[A_index]
        tree = DT(A)

        print("Most gains: ", A)

        new_attr = [x for x in attributes if x != A]

        for v_k in examples[A].unique():
            exs = examples[examples[A] == v_k]
            subtree = DTL(exs, new_attr, examples)
            tree.labels[v_k] = subtree

        return tree

attributes = list(training_data.columns.values)
attributes.remove('Survived')
attributes.remove('Name')
attributes.remove('Age')
attributes.remove('Cabin')
attributes.remove('Ticket')
attributes.remove('Fare')




DT = DTL(training_data, attributes, [])
print(DT)