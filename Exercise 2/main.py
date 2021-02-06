import numpy as np

x_0 = np.array([0.5, 0.5])
T = np.array([[0.8, 0.3],[0.2, 0.7]])
O = [np.array([[0.75, 0], [0, 0.2]]), np.array([[0.25, 0], [0, 0.8]])]

# Umbrella example from book
x_0 = np.array([0.5, 0.5])
T = np.array([[0.7, 0.3],[0.3, 0.7]])
O = np.array([np.array([[0.9, 0], [0, 0.2]]), np.array([[0.1, 0], [0, 0.8]])])


evidence = np.array([0, 0, 1, 0, 1, 0]) # 0 = birds nearby, 1 = no birds nearby

# b) Filtering:

def forward(t):
    if (t > len(evidence)):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 0):
        return x_0
    f = O[evidence[t-1]] @ (T.transpose() @ forward(t-1))
    return f/np.sum(f)

""" for i in range(1, 6):
    print(f"Filtering t={i}: {forward(i)}") """

# c) Prediction:

def predict(t, k):
    if (k == 0):
        return forward(t)

    p = T.transpose() @ predict(t, k-1)
    return p/np.sum(p)

""" for i in range(7, 31):
    print(f"Prection t={i}: {predict(6, i-6)}") """
# As t increases it goes towards [0.5, 0.5], which is to be expected as over time the evidence gives us very little and therefore it should go back
# to the initial distribution

# d) Smoothing:


def backward(b, ev):
    sum_xk_1 = np.zeros(2)
    for idx,p in enumerate(b):
        sum_xk_1 += sensor_model[ev][idx]*p*transition_model[idx]
    return sum_xk_1

def forward_backward(evidence):
    t = len(evidence)
    fv = np.zeros((t+1, 2))
    b = np.ones(2)
    sv = np.zeros((t, 2))

    fv[0] = x_0
    for i in range(1, t+1):
        fv[i] = forward(i)
    for i in range(t, 0, -1):
        sv[i-1] = (fv[i]*b)/np.sum(fv[i]*b)
        b = backward(b, evidence[i-1])
    return sv

for i in range(6):
    print("Smoothed result: ", forward_backward(evidence[:i]))

# e) Most likely sequence:

def viterbi(t):
    if (t > len(evidence)):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 0):
        return x_0

    sum_xt = 0
    f_t_1 = forward(t-1)
    for idx,p in enumerate(f_t_1):
        sum_xt += transition_model[idx]*p
    f = sensor_model[evidence[t-1]]*sum_xt
    return f/np.sum(f)

print("Most likely sequence: ")
