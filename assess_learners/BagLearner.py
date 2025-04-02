import numpy as np
from numpy.ma.core import right_shift

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt

class BagLearner(object):
    """
    This is a Bag Learner

    """

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """
        Constructor method
        """
        self.bags = bags
        self.models = []
        for i in np.arange(self.bags):
            self.models.append(learner(**kwargs))
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """
        Returns the author
        """
        return "sphadnis9"

    def add_evidence(self, data_x, data_y):
        """
        Builds the decision tree
        """
        data = np.column_stack((data_x, data_y))
        bag_data_index = self.get_data_index_for_bags(data)

        for i in np.arange(self.bags):
            self.models[i].add_evidence(data[bag_data_index[i, :], 0:-1], data[bag_data_index[i, :], -1])

    def query(self, points):
        """
        Create bag learner
        """
        ans = np.empty((self.bags, points.shape[0]))
        for i in np.arange(self.bags):
            ans[i] = self.models[i].query(points)

        return np.mean(ans, axis=0)

    def get_data_index_for_bags(self, data):
        return np.random.choice(np.arange(data.shape[0]), size=(self.bags, data.shape[0]), replace=True)

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
