from probability import *

def viterbi(HMM, ev):
    """
    [Equation 15.11]
    Viterbi algorithm to find the most likely sequence. Computes the best path and the
    corresponding probabilities, given an HMM model and a sequence of observations."""
    t = len(ev)
    ev = ev.copy()
    ev.insert(0, None)

    m = [[0.0, 0.0] for _ in range(len(ev) - 1)]

    # the recursion is initialized with m1 = forward(P(X0), e1)
    m[0] = forward(HMM, HMM.prior, ev[1])
    # keep track of maximizing predecessors
    backtracking_graph = []

    for i in range(1, t):
        m[i] = element_wise_product(HMM.sensor_dist(ev[i + 1]),
                                    [max(element_wise_product(HMM.transition_model[0], m[i - 1])),
                                     max(element_wise_product(HMM.transition_model[1], m[i - 1]))])
        backtracking_graph.append([np.argmax(element_wise_product(HMM.transition_model[0], m[i - 1])),
                                   np.argmax(element_wise_product(HMM.transition_model[1], m[i - 1]))])

    # computed probabilities
    ml_probabilities = [0.0] * (len(ev) - 1)
    # most likely sequence
    ml_path = [True] * (len(ev) - 1)

    # the construction of the most likely sequence starts in the final state with the largest probability, and
    # runs backwards; the algorithm needs to store for each xt its predecessor xt-1 maximizing its probability
    i_max = np.argmax(m[-1])

    for i in range(t - 1, -1, -1):
        ml_probabilities[i] = m[i][i_max]
        ml_path[i] = True if i_max == 0 else False
        if i > 0:
            i_max = backtracking_graph[i - 1][i_max]

    return ml_path, ml_probabilities

umbrella_transition = [[0.8, 0.2], [0.3, 0.7]]
umbrella_sensor = [[0.75, 0.2], [0.25, 0.8]]
umbrellaHMM = HiddenMarkovModel(umbrella_transition, umbrella_sensor)

umbrella_evidence = [True, True, False, True, False, True]

viterbi(umbrellaHMM, umbrella_evidence)