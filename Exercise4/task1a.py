import pandas as pd
import numpy as np

training_data = pd.read_csv('Exercise4/train.csv')
#training_data.value_counts()
#print(training_data['Survived'].nunique())
#print(len(training_data['Survived']))

print()
def gain(A):

    pass

def remainder(A, examples):

    n_p_sum = len(examples['Survived'].size())

    s = 0

    for i in range(examples[A].nunique()):
        pi = examples.groupby[A][i]
        ni = 0
    return s/n_p_sum

def H(p):

    s = 0
    for i in range(len(p)):
        s -= p[i]*np.log2(p[i])

    return s

""" def DTL(examples, attributes, parent_examples):

    if not examples:
        return parent_examples
    elif check_same_class(examples):
        return class
    elif not attributes:
        return Mode(examples)
    else:
        recursive shit """

#DT = DTL(training_data, todo, todo)