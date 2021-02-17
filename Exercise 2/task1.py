import numpy as np
import matplotlib.pyplot as plt

x_0 = np.array([0.5, 0.5])
T = np.array([[0.8, 0.2],[0.3, 0.7]])
O = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])]
evidence = np.array([None, 0, 0, 1, 0, 1, 0]) # 0 = birds nearby, 1 = no birds nearby

N = len(evidence)-1 # correct length of evidence
# Umbrella example from book
#x_0 = np.array([0.5, 0.5])
#T = np.array([[0.7, 0.3],[0.3, 0.7]])
#O = np.array([np.array([[0.9, 0], [0, 0.2]]), np.array([[0.1, 0], [0, 0.8]])])
#evidence =  np.array([None, 0, 0, 1, 0, 0])

# b) Filtering:

def forward():
    f = np.zeros((N+1, 2))
    f[0] = x_0 # filtering at t=0 is just the initial distribution


    for i in range(1, N+1):
        f[i] = O[evidence[i]] @ T.transpose() @ f[i-1] # implement equation 15.12 from AIMA (Artifical Intelligence - A Modern Approach)
        f[i] = f[i]/np.sum(f[i]) # normalize
    return f
# c) Prediction:

def predict(t, k):

    # base case in recursion is just filtering on last step
    if (k == 0):
        return forward(t)

    # use equation 15.12 from AIMA, but without evidence
    p = T.transpose() @ predict(t, k-1)
    return p/np.sum(p) # normalize vector

# d) Smoothing:

def backward_hmm(b, ev):
    if ev == None:
        return 0

    temp = T @ O[ev] @ b
    return temp

def forward_backward(evidence):
    t = len(evidence)-1
    sv = np.zeros((len(evidence), 2))
    b = np.ones(2)
    for i in range(t, -1, -1):
        f = forward(i)
        sv[i] = (f * b)/np.sum(f * b)
        b = backward_hmm(b, evidence[i])
    return sv



# e) Most likely sequence:

def viterbi2():
    K = len(evidence)-1
    N = 2
    T_1 = np.zeros((N, K))
    T_2 = np.zeros((N, K))

    T_1[:, 0] = forward()[1]
    for j in range(1, len(evidence)-1):
        p = T_1[:, j-1]*(O[evidence[j+1]] @ T.transpose())
        T_1[:,j] = np.max(p, 1)
        T_2[:,j] = np.argmax(p, 1)

    sequence = np.zeros(K)
    sequence[K-1] = np.argmax(T_1[:, K-1])

    for j in range(K-1, 0, -1):
        sequence[j-1] = T_2[int(sequence[j]), j]


    return sequence



print(viterbi2())

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
    for i in t:
        p[i-7] = predict(6, i-6)[0]

    plt.bar(t, p, label=r"$P(x_t | e_{1:6})$")
    plt.title("Predicted result")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.show()

def plot_smoothing():
    t = np.array([x for x in range(0, 6)])
    p = np.round(forward_backward(evidence)[0:6:, 0],3)
    plt.title("Smoothed result")
    plt.bar(t, p, label=r"$P(x_t = FishNearby | e_{1:6})$")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend()
    xlocs=[i for i in range(0,6)]
    for i, v in enumerate(p):
        plt.text(xlocs[i]-0.25, v-0.05, str(v), color="white")
    plt.show()

plot_filtering()
#plot_prediction()
#plot_smoothing()