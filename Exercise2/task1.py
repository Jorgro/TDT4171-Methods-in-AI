import numpy as np
import matplotlib.pyplot as plt

x_0 = np.array([0.5, 0.5])
T = np.array([[0.8, 0.2],[0.3, 0.7]])
O = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])]
evidence = np.array([None, 0, 0, 1, 0, 1, 0]) # 0 = birds nearby, 1 = no birds nearby, None at start to simplify indexing

# Umbrella example from book used for debugging
#x_0 = np.array([0.5, 0.5])
#T = np.array([[0.7, 0.3],[0.3, 0.7]])
#O = np.array([np.array([[0.9, 0], [0, 0.2]]), np.array([[0.1, 0], [0, 0.8]])])
#evidence =  np.array([None, 0, 0, 1, 0, 0])

N = len(evidence)-1 # correct length of evidence, minus 1 since we added None at the start to simplify the indexing

# b) Filtering:

def forward():
    """
    Implements the forward equation 15.12 from AIMA with evidence, will only filter as long as there is evidence

    Returns:
        Numpy array of the filtered distribution of state at time = index
    """
    f = np.zeros((N+1, 2))
    f[0] = x_0 # filtering at t=0 is just the initial distribution


    for i in range(1, N+1):
        f[i] = O[evidence[i]] @ T.transpose() @ f[i-1] # implement equation 15.12 from AIMA (Artifical Intelligence - A Modern Approach)
        f[i] = f[i]/np.sum(f[i]) # normalize
    return f

# c) Prediction:

def predict(k):
    """
    Implements the forward equation 15.12 from AIMA without any new evidence

    Args:
        k: How far it should predict into the future

    Returns:
        Numpy array of the predicted distribution of state at time = index
    """

    f = np.zeros((k+1, 2))
    f[0] = forward()[-1] # using the latest filtering (with evidence) as start

    for i in range(1, k+1):
        f[i] = T.transpose() @ f[i-1] # forward update as in equation 15.12, but without any evidence as we are predicting
        f[i] = f[i]/np.sum(f[i]) # normalize
    return f

# d) Smoothing:

def backward(b, ev):
    """
    Implements the backward equation 15.13 from AIMA

    Args:
        b: The previous backward value
        ev: The evidence at this time step

    Returns:
        Numpy array (vector) for this backward value
    """
    # Used to handle the case if evidence is equal to None, which will happen at the last iteration in forward-backward
    # the return of 0 doesn't matter as the backward value is not used for this case
    if ev == None:
        return 0

    return  T @ O[ev] @ b # backward update as in equation 15.13

def forward_backward():
    """
    Implements the forward-backward algoruthm from AIMA

    Returns:
        Numpy array of the smoothed distributions
    """

    sv = np.zeros((N+1, 2))
    b = np.ones(2)
    f = forward() # get the forward values
    for i in range(N, -1, -1):
        sv[i] = (f[i] * b)/np.sum(f[i] * b) # calculate the smoothed distrubution  from equation 15.8
        b = backward(b, evidence[i]) # calculate new backward value
    return sv



# e) Most likely sequence:

def viterbi():
    """
    Implementation of the Viterbi algorithm
    This implementation uses the pseudocode from wikipedia (https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode)
    with some simplifications using matrices in numpy

    Returns:
        The most likely sequence of states for the evidence
    """

    K = len(evidence)-1 # number of evidence given
    N = 2 # number of states
    T_1 = np.zeros((N, K)) # stores the probabilities of most likely paths so far (not normalized except for first element)
    T_2 = np.zeros((N, K)) # stores the most likely path so far to the state

    T_1[:, 0] = forward()[1] # first element is just the forward for t=1

    # go through evidence
    for j in range(1, len(evidence)-1):
        p = T_1[:, j-1]*(O[evidence[j+1]] @ T.transpose()) # implement equation 15.11 without maximization
        T_1[:,j] = np.max(p, 1) # find the max probability
        T_2[:,j] = np.argmax(p, 1) # find the state that maximizes this probability

    sequence = np.zeros(K) # used to store the sequence of states
    sequence[K-1] = np.argmax(T_1[:, K-1]) # find the most like state at the end

    # go backward from the most likely state at the end to find the path
    for j in range(K-1, 0, -1):
        sequence[j-1] = T_2[int(sequence[j]), j] # find the path using T_2 which has the most likely path to that state

    print("T_1: ", T_1)
    print("T_2: ", T_2)
    return sequence

# The following functions are used to plot the results

def plot_filtering():
    t = np.array([x for x in range(1, 7)])
    p = forward()[1:7,0]
    p = np.round(p, 3)
    plt.title("Filtering result")
    plt.bar(t, p, label=r"$P(x_t = FishNearby | e_{1:t})$")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend()
    xlocs=[i for i in range(1,7)]
    for i, v in enumerate(p):
        plt.text(xlocs[i]-0.25, v-0.05, str(v), color="white")

    plt.show()

def plot_prediction():
    t = np.array([x for x in range(7, 31)])
    p = np.zeros(31-7)
    p = predict(30)[1:25,0]
    plt.bar(t, p, label=r"$P(x_t | e_{1:6})$")
    plt.title("Predicted result")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.show()

def plot_smoothing():
    t = np.array([x for x in range(0, 6)])
    p = np.round(forward_backward()[0:6:, 0],3)
    plt.title("Smoothed result")
    plt.bar(t, p, label=r"$P(x_t = FishNearby | e_{1:6})$")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend()
    xlocs=[i for i in range(0,6)]
    for i, v in enumerate(p):
        plt.text(xlocs[i]-0.25, v-0.05, str(v), color="white")
    plt.show()

if __name__ == '__main__':
    plot_filtering()
    plot_prediction()
    plot_smoothing()
    print("Most likely sequence for the evidence given: ", viterbi())