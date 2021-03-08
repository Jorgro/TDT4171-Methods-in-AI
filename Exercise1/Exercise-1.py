from collections import defaultdict

import numpy as np


class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        both with 2 possible states.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(0) | cond0(1) | cond0(0) | cond0(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[2, 2])
        """
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError(
                "Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError(
                "Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + \
                '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + \
                '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def __repr__(self):
        """
        Better representation of object when printing a list
        """
        return "Node " + self.name

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(
                f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(
                f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(
                f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(
                f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(
                    f"Variable {variable.name} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * \
                np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """

    def __init__(self):
        # All nodes start out with 0 edges
        self.edges = defaultdict(lambda: [])
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError(
                "Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError(
                "Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        Kahn's algorithm for putting variables in lexicographical topological order

        Returns:
            List of sorted variables
        """
        L = [] # List with topological order
        S = [] # List of nodes with no incoming edges

        edges_copy = self.edges.copy() # using a copy of the edges to not ruin the BN

        # We start off with the nodes without any incoming edges
        for node in self.variables.values():
            if not node.parents:
                S.append(node)

        # Count number of visited nodes
        count = 0
        while S:
            count += 1
            n = S.pop()
            L.append(n) # Add it to list since it doesn't have any incoming edges

            # Go through all edges from n
            while edges_copy[n]:
                    child = edges_copy[n].pop() # remove edge from n -> child
                    # See if m still has incoming edges
                    child_has_incoming_edge = False
                    for node in self.variables.values():
                        if child in edges_copy[node]:
                            child_has_incoming_edge = True
                    # If it doesn't have any incoming edges we add it to the queue
                    if not child_has_incoming_edge:
                        S.append(child)

        # If number of visited nodes is not equal to the number of variables in graph
        # it is not possible to topological sort the variables
        if count != len(self.variables.values()):
            raise RuntimeError("Cycle identified! This is not a DAG and can't be sorted.")
        return L


class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X, evidence):
        # Initialize q as a vector with as many states as the variable asked for
        q = np.zeros(self.bayesian_network.variables[X].no_states)

        # Go through each possible state and find the probability by using inference by enumeration
        for i in range(self.bayesian_network.variables[X].no_states):
            e = evidence.copy() # Copy evidence to make sure each _enumerate_all call starts with the original evidence variables
            e[X] = i # Set the evidence of X to the state iterated
            q[i] = self._enumerate_all(self.topo_order.copy(), e)

        # Divide by the sum as the elements in q should sum to 1, this is due to using a normalization factor
        return q/(np.sum(q))

    def _enumerate_all(self, variables, evidence):
        if not variables:
            return 1.0

        Y = variables[0] # Y is the first

        v = variables.copy()
        v.remove(Y)
        if Y.name in evidence.keys():
            # Calculate the probability with the assignment of Y
            e = evidence.copy()
            return Y.probability(e[Y.name], e)*self._enumerate_all(v, e)
        else:
            # Calculate the sum over all possible assignments of Y
            sum_over_y = 0
            for i in range(Y.no_states):
                e = evidence.copy()
                e[Y.name] = i
                sum_over_y += Y.probability(e[Y.name], e)*self._enumerate_all(v, e)
            return sum_over_y


    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1)
        return Variable(var, self.bayesian_network.variables[var].no_states, q)

def monty_hall():
    chosen_by_guest = Variable('A', 3, [[1/3], [1/3], [1/3]]) # The guest's first choice is chosen randomly between the 3 doors
    prize = Variable('B', 3, [[1/3], [1/3], [1/3]]) # The prize is chosen randomly between the 3 doors
    opened_by_host = Variable('C', 3,
                        [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5],
                        [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
                        [0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]],
                  parents=['A', 'B'],
                  no_parent_states=[3, 3])
     # The host can never open the door with the prize or the one chosen by the guest, so these will always have a probability of 0
     # The probability table for opened by host was set up using the following two cases:
     # 1. If the same door was chosen by both guest and prize the host has two doors to chose from, giving prbability of 0.5 on each of them.
     # 2. If different doors were chosen by the guest and prize then the host only has one door to choose from, giving probability of 1 on this door.

    print(f"Probability distribution, P({chosen_by_guest.name})")
    print(chosen_by_guest)

    print(f"Probability distribution, P({prize.name})")
    print(prize)

    print(f"Probability distribution, P({opened_by_host.name} | {prize.name}, {chosen_by_guest.name})")
    print(opened_by_host)

    # Set up the bayesian network according to the figure 4 in the pdf
    bn = BayesianNetwork()
    bn.add_variable(prize)
    bn.add_variable(opened_by_host)
    bn.add_variable(chosen_by_guest)
    bn.add_edge(prize, opened_by_host)
    bn.add_edge(chosen_by_guest, opened_by_host)

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('B', {'A': 0, 'C': 2}) # Using 0 indexing on the doors => door 0 => door 1 and door 2 => door 3

    print(f"Probability distribution, P({prize.name} | {chosen_by_guest.name}, {opened_by_host.name})")
    print(posterior)

def exercise3():
    B = Variable('A', 2, [[0.5], [0.5]])
    M = Variable('B', 2, [[0.9, 0.65], [0.1, 0.35]], parents=['A'], no_parent_states=[2])
    P = Variable('C', 2, [[0.9, 0.4, 0.7, 0.2], [0.1, 0.6, 0.3, 0.8]], parents=['B', 'A'], no_parent_states=[2, 2])

    bn = BayesianNetwork()
    bn.add_variable(M)
    bn.add_variable(P)
    bn.add_variable(B)
    bn.add_edge(B, M)
    bn.add_edge(B, P)
    bn.add_edge(M, P)
    inference = InferenceByEnumeration(bn)
    posterior = inference.query('C', {'A': 1})
    print(posterior)
if __name__ == '__main__':
    #monty_hall()
    exercise3()