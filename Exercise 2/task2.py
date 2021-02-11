import numpy as np

x_0 = np.array([0.7, 0.3])
T = np.array([[0.8, 0.2],[0.3, 0.7]])

at = np.array([[0.7, 0.3], [0.2, 0.8]])
fg = np.array([[0.3, 0.7], [0.1, 0.9]])
O = [[
    np.array([
        [at[0][0]*fg[0][0], 0],
        [0, at[1][0]*fg[1][0]]]),
    np.array([
        [at[0][0]*fg[0][1], 0],
        [0, at[1][0]*fg[1][1]]])
],[
    np.array([
        [at[0][1]*fg[0][0], 0],
        [0, at[1][1]*fg[1][0]]]),
    np.array([
        [at[0][1]*fg[0][1], 0],
        [0, at[1][1]*fg[1][1]]])
]]
evidence = np.array([[None, None], [0, 0], [1, 0], [1, 1], [0, 1]])

# Wrote it as an HMM since easier to use the algorithms already made, otherwise we would have to "unroll" the DBN and use the algorithm created in exercise 1

def forward(t):
    if (t > len(evidence)-1):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 0):
        return x_0
    #print("O: ", O[evidence[t][0]][evidence[t][1]])
    #print("evidence: ", evidence[t])
    f = O[evidence[t][0]][evidence[t][1]] @ (T.transpose() @ forward(t-1))
    return f/np.sum(f)


def predict(t, k):
    if (k == 0):
        return forward(t)

    p = T.transpose() @ predict(t, k-1)
    return p/np.sum(p)


def backward_hmm(b, ev):
    print(O[ev[0]][ev[1]])
    return T @ O[ev[0]][ev[1]] @ b

def forward_backward(evidence):
    t = len(evidence)-1
    sv = np.zeros((t, 2))
    b = np.ones(2)

    for i in range(t, 0, -1):
        b = backward_hmm(b, evidence[i])
        f = forward(i-1)
        sv[i-1] = (f * b)/np.sum(f * b)
    return sv
# b)

for i in range(0, 5):
    print(f"Forward t={i}: {forward(i)}")
print()
# c)
for i in range(5, 9):
    print(f"Predict t={i}: {predict(4, i-4)}")
print()

print(f"Smoothed: {forward_backward(evidence)}")


