import numpy as np
import matplotlib.pyplot as plt

x_0 = np.array([0.5, 0.5])
T = np.array([[0.8, 0.2],[0.3, 0.7]])
O = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])]
evidence = np.array([None, 0, 0, 1, 0, 1, 0]) # 0 = birds nearby, 1 = no birds nearby

# Umbrella example from book
#x_0 = np.array([0.5, 0.5])
#T = np.array([[0.7, 0.3],[0.3, 0.7]])
#O = np.array([np.array([[0.9, 0], [0, 0.2]]), np.array([[0.1, 0], [0, 0.8]])])
#evidence =  np.array([None, 0, 0, 1, 0, 0])


# b) Filtering:

def forward(t):
    if (t > len(evidence)-1):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 0):
        return x_0
    f = O[evidence[t]] @ (T.transpose() @ forward(t-1))
    return f/np.sum(f)

# c) Prediction:

def predict(t, k):
    if (k == 0):
        return forward(t)

    p = T.transpose() @ predict(t, k-1)
    return p/np.sum(p)

# d) Smoothing:

def backward_hmm(b, ev):
    return T @ O[ev] @ b

def forward_backward(evidence):
    t = len(evidence)-1
    f = forward(t)
    sv = np.zeros((t, 2))
    b = np.ones(2)

    for i in range(t, 0, -1):
        sv[i-1] = (f * b)/np.sum(f * b)
        b = backward_hmm(b, evidence[i])
        t_f = np.linalg.inv(T.transpose()) @ np.linalg.inv(O[evidence[i]]) @ f
        f = t_f/np.sum(t_f)
    return sv

def plot_result():

    t = np.array([x for x in range(1, 7)])
    p = np.zeros(6)
    for i in t:
        p[i-1] = forward(i)[0]
    plt.subplot(221)
    plt.bar(t, p, label=r"$P(x_t | e_{1:t})$")



    t = np.array([x for x in range(7, 31)])
    p = np.zeros(31-7)
    for i in t:
        p[i-7] = predict(6, i-6)[0]

    plt.bar(t, p, label=r"$P(x_t | e_{1:6})$")
    plt.title("Filtered and predicted result")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("probability")

    # plot filtered
    t = np.array([x for x in range(0, 6)])
    p = np.zeros(6)
    p[0] = x_0[0]
    for i in t:
        p[i] = forward(i)[0]
    plt.subplot(223)
    plt.title("Filtering result")
    plt.bar(t, p, label=r"$P(x_t | e_{1:t})$")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend()

    # plot
    t = np.array([x for x in range(0, 6)])
    p = forward_backward(evidence)[::, 0]
    plt.subplot(224)
    plt.title("Smoothed result")
    plt.bar(t, p, label=r"$P(x_t | e_{1:6})$")
    plt.xlabel("t")
    plt.ylabel("probability")
    plt.legend()
    plt.show()



plot_result()
# e) Most likely sequence:

# 0.8182*0.3*0.2 = 0.04909
# 0.8182*0.7*0.9 = 0.5155



def viterbi(t):
    if (t > len(evidence)):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 1):
        f = forward(1)
        sequence.append(np.argmax(f))
        probabilities.append(f[np.argmax(f)])
        return forward(1)

    m = viterbi(t-1)
    xt_max = np.argmax(m)
    k = m[xt_max] * (T @ O[evidence[t]])[xt_max]
    sequence.append(np.argmax(k))
    probabilities.append(k[np.argmax(k)])
    return k

probabilities = []
sequence = []

#viterbi(5)
#print("Sequence: ", sequence)
#print("probabilities: ", np.round(probabilities, 5))
