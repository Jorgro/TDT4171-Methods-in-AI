import numpy as np

x_0 = np.array([0.5, 0.5])
transition_model = np.array([[0.8, 0.2],[0.3, 0.7]])
sensor_model = np.array([[0.75, 0.25],[0.2, 0.8]])

e = np.array([0, 0, 1, 0, 1, 0]) # 0 = birds nearby, 1 = no birds nearby

# b) Filtering:

def forward(t):
    if (t > len(e)):
        raise ValueError("t can't be larger than the evidence provided!") # TODO: Is this correct?
    if (t == 0):
        return x_0
    # alpha * sensormodel(t+1)*sum_xt (transitionmodel_t * forward(t-1))
    sum_xt = 0
    for idx,p in enumerate(forward(t-1)):
        sum_xt += transition_model[idx]*p
    f = sensor_model[e[t-1]]*sum_xt
    return f/np.sum(f)

print("Filtering t=6: ", forward(6))



# c) Prediction:

# d) Smoothing:

# e) Most likely sequence: