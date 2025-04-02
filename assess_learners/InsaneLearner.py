import numpy as np
import LinRegLearner as lrl
import RTLearner as rt
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learner = []
        for i in range(0,20):
            self.learner.append(bl.BagLearner(rt.RTLearner, kwargs={"verbose": self.verbose}, bags=20))

    def author(self): return "sphadnis9"

    def add_evidence(self, data_x, data_y):
        for bag_learner in self.learner:
            bag_learner.add_evidence(data_x, data_y)

    def query(self, points):
        ans = np.empty((20, points.shape[0]))
        for i in range(0,20): ans[i] = self.learner[i].query(points)
        return np.mean(ans, axis=0)
