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
        print(curr_pro)
        for i in range(1, len(obs)):
            last_pro = curr_pro
            curr_pro = {}
            for curr_state in states:
                max_pro, last_sta = max(((last_pro[last_state]*t_pro[last_state][curr_state]*e_pro[curr_state][obs[i]], last_state)
                                        for last_state in states))
                curr_pro[curr_state] = max_pro
                print(max_pro)
                path[curr_state].append(last_sta)
        max_pro = -1
        max_path = None
        for s in states:
            path[s].append(s)
            if curr_pro[s] > max_pro:
                max_path = path[s]
                max_pro = curr_pro[s]
            print('%s: %s'%(curr_pro[s], path[s])) # different path and their probability

        return max_path


if __name__ == '__main__':
	obs = ['normal', 'normal', 'cold', 'normal', 'cold', 'normal']
	print(Viterbit(obs, states, start_probability, transition_probability, emission_probability))