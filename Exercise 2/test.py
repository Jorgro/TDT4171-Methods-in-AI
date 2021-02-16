# -*- coding: utf-8 -*-
# @Author: WuLC
# @Date:   2017-04-02 08:52:24
# @Last Modified by:   WuLC
# @Last Modified time: 2017-04-02 09:50:31

###########################################################################################################
# Viterbi Algorithm for HMM
# dp, time complexity O(mn^2), m is the length of sequence of observation, n is the number of hidden states
##########################################################################################################


# five elements for HMM
states = ('Healthy', 'Fever')

observations = ('normal', 'cold')

start_probability = {'Healthy': 0.5, 'Fever': 0.5}

transition_probability = {
   'Healthy' : {'Healthy': 0.8, 'Fever': 0.2},
   'Fever' :   {'Healthy': 0.3, 'Fever': 0.7},
   }

emission_probability = {
   'Healthy' : {'normal': 0.75, 'cold': 0.25},
   'Fever'   : {'normal': 0.2, 'cold': 0.8},
   }



def Viterbit(obs, states, s_pro, t_pro, e_pro):
        path = { s:[] for s in states} # init path: path[s] represents the path ends with s
        curr_pro = {}
        for s in states:
            curr_pro[s] = s_pro[s]*e_pro[s][obs[0]]
        for i in range(1, len(obs)):
            last_pro = curr_pro
            curr_pro = {}
            for curr_state in states:
                max_pro, last_sta = max(((last_pro[last_state]*t_pro[last_state][curr_state]*e_pro[curr_state][obs[i]], last_state)
                                        for last_state in states))
                curr_pro[curr_state] = max_pro
                path[curr_state].append(last_sta)
        max_pro = -1
        max_path = None
        for s in states:
            path[s].append(s)
            if curr_pro[s] > max_pro:
                max_path = path[s]
                max_pro = curr_pro[s]
            print('%s: %s'%(curr_pro[s], path[s])) # different path and their probabilityf

        return max_path

def viterbi1(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            max_prob = max_tr_prob * emit_p[st][obs[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    best_st = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ("The steps of states are " + " ".join(opt) + " with highest probability of %s" % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


import numpy as np

def viterbi(y, A, B, Pi=None):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2

import operator

class HMM():

    def __init__(self) -> None:
        self.transition = np.array([[0.8, 0.2], # A = transition probs. / 2 states
                           [0.3, 0.7]])
        self.priors = np.array([0.5, 0.5])
        self.emission = np.array([[0.75, 0.25], # B = emission (observation) probs. / 3 obs modes
                         [0.2, 0.8]])

    def viterbi2(self,observations):
        """Return the best path, given an HMM model and a sequence of observations"""
        # A - initialise stuff
        nSamples = len(observations)
        nStates = self.transition.shape[0] # number of states
        c = np.zeros(nSamples) #scale factors (necessary to prevent underflow)
        viterbi = np.zeros((nStates,nSamples)) # initialise viterbi table
        psi = np.zeros((nStates,nSamples)) # initialise the best path table
        best_path = np.zeros(nSamples); # this will be your output

        # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
        viterbi[:,0] = self.priors.T * self.emission[:,observations[0]]
        print("priors: ", self.priors.T)
        print("observation: ", self.emission[:,observations[0]])
        print("vit: ", viterbi)
        c[0] = 1.0/np.sum(viterbi[:,0])
        viterbi[:,0] = c[0] * viterbi[:,0] # apply the scaling factor
        psi[0] = 0;

        # C- Do the iterations for viterbi and psi for time>0 until T
        for t in range(1,nSamples): # loop through time
            for s in range (0,nStates): # loop through the states @(t-1)
                trans_p = viterbi[:,t-1] * self.transition[:,s]
                psi[s,t], viterbi[s,t] = max(enumerate(trans_p), key=operator.itemgetter(1))
                viterbi[s,t] = viterbi[s,t]*self.emission[s,observations[t]]
            c[t] = 1.0/np.sum(viterbi[:,t]) # scaling factor
            viterbi[:,t] = c[t] * viterbi[:,t]

        print("Viterbi: ", viterbi)
        # D - Back-tracking
        best_path[nSamples-1] =  viterbi[:,nSamples-1].argmax() # last state
        print("best path: ", best_path)
        for t in range(nSamples-1,0,-1): # states of (last-1)th to 0th time step
            best_path[t-1] = psi[int(best_path[t]),t]

        return best_path

if __name__ == '__main__':
	#obs = ['normal', 'normal', 'cold', 'normal', 'cold', 'normal']
	#print(viterbi1(obs, states, start_probability, transition_probability, emission_probability))
    hmm = HMM()
    print(hmm.viterbi2([0, 0, 1, 0, 1, 0]))

   # x, T1, T2 = viterbi(np.array([0, 0, 1, 0, 1, 0]), np.array([[0.8, 0.2],[0.3, 0.7]]), np.array([]))