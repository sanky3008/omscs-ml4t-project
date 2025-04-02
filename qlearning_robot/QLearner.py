""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Sankalp Phadnis	 	 	 			  		 			     			  	 
GT User ID: sphadnis9	  	   		 	 	 			  		 			     			  	 
GT ID: 904081199 		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import random as rand  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np
  		  	   		 	 	 			  		 			     			  	 
class QLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This is a Q learner object.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		 	 	 			  		 			     			  	 
    :type num_states: int  		  	   		 	 	 			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		 	 	 			  		 			     			  	 
    :type num_actions: int  		  	   		 	 	 			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type alpha: float  		  	   		 	 	 			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type gamma: float  		  	   		 	 	 			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type rar: float  		  	   		 	 	 			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	 	 			  		 			     			  	 
    :type radr: float  		  	   		 	 	 			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	 	 			  		 			     			  	 
    :type dyna: int  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    def __init__(  		  	   		 	 	 			  		 			     			  	 
        self,  		  	   		 	 	 			  		 			     			  	 
        num_states=100,  		  	   		 	 	 			  		 			     			  	 
        num_actions=4,  		  	   		 	 	 			  		 			     			  	 
        alpha=0.2,  		  	   		 	 	 			  		 			     			  	 
        gamma=0.9,  		  	   		 	 	 			  		 			     			  	 
        rar=0.5,  		  	   		 	 	 			  		 			     			  	 
        radr=0.99,  		  	   		 	 	 			  		 			     			  	 
        dyna=0,  		  	   		 	 	 			  		 			     			  	 
        verbose=False,  		  	   		 	 	 			  		 			     			  	 
    ):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.verbose = verbose  		  	   		 	 	 			  		 			     			  	 
        self.num_actions = num_actions
        self.q = np.zeros((num_states, num_actions))
        self.s = 0
        self.a = 0
        # self.exp = np.empty((0,4))
        self.exp = dict() # keeping it as dict to avoid using np.unique, which takes time
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
  		  	   		 	 	 			  		 			     			  	 
    def querysetstate(self, s):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param s: The new state  		  	   		 	 	 			  		 			     			  	 
        :type s: int  		  	   		 	 	 			  		 			     			  	 
        :return: The selected action  		  	   		 	 	 			  		 			     			  	 
        :rtype: int  		  	   		 	 	 			  		 			     			  	 
        """
        # get the new action based on greedy approach
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.q[s].argmax()

        # update current variables
        self.rar = self.rar * self.radr
        self.s = s
        self.a = action

        return action  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    def query(self, s_prime, r):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Update the Q table and return an action  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param s_prime: The new state  		  	   		 	 	 			  		 			     			  	 
        :type s_prime: int  		  	   		 	 	 			  		 			     			  	 
        :param r: The immediate reward  		  	   		 	 	 			  		 			     			  	 
        :type r: float  		  	   		 	 	 			  		 			     			  	 
        :return: The selected action  		  	   		 	 	 			  		 			     			  	 
        :rtype: int  		  	   		 	 	 			  		 			     			  	 
        """
        s = self.s
        a = self.a

        # get the new action based on greedy approach
        if rand.uniform(0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.q[s_prime].argmax()

        # update the q-table
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * self.q[s_prime].max())

        # dyna-q:
        # In this, about dyna samples are taken from experience and replayed to help make the propagation of rewards faster
        if self.dyna > 0:
            # update experience. I have kept this inside dyna condition for we don't need to maintain this array if dyna = 0, saving time.
            if (self.s, self.a) not in self.exp:
                self.exp[(self.s, self.a)] = [(s_prime, r)]
            else:
                self.exp[(self.s, self.a)].append((s_prime, r))

            # generate stochastic samples for dyna-q.
            # if len(self.exp.keys()) >= self.dyna: # we will only do experience replay if we have enough samples, https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits
            sample_keys = rand.choices(list(self.exp.keys()), k=self.dyna)
            for key in sample_keys:
                s_dyna = key[0]
                a_dyna = key[1]
                s_prime_dyna, r_dyna = rand.choice(self.exp[key])

                # update the Q-table
                self.q[s_dyna, a_dyna] = (1 - self.alpha) * self.q[s_dyna, a_dyna] + self.alpha * (r_dyna + self.gamma * self.q[s_prime_dyna].max())

        # update current variables
        self.rar = self.rar*self.radr
        self.s = s_prime
        self.a = action

        return action

    def author(self):
        return "sphadnis9"

    def study_group(self):
        return "sphadnis9"
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		 	 	 			  		 			     			  	 
