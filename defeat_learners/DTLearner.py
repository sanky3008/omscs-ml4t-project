import numpy as np
from numpy.ma.core import right_shift


class DTLearner(object):
    """
    This is a Deterministic Decision Tree Learner

    """

    def __init__(self, verbose=False, leaf_size=1):
        """
        Constructor method
        """
        self.tree = np.array([])
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.node_count = 0

    def author(self):
        """
        Returns the author
        """
        return "sphadnis9"

    def add_evidence(self, data_x, data_y):
        """
        Builds the decision tree
        """
        if self.verbose:
            print("Building tree...")

        data = np.column_stack((data_x, data_y)) # recreate the data https://www.geeksforgeeks.org/python-ways-to-add-row-columns-in-numpy-array/
        self.tree = self.build_tree(data)

        if self.verbose:
            print("Final Tree:\n")
            print(self.tree)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        pred_y = np.array([])
        for x in points:
            pointer = 0
            tree = self.tree

            if len(tree.shape) == 1 and tree[0] == -1.0:
                return tree[1]
            elif len(tree.shape) == 1 and tree[0] != -1.0:
                print("ERROR")
                return

            while tree[pointer, 0] != -1.0:
                if x[int(tree[pointer, 0])] <= tree[pointer, 1]:
                    pointer += int(tree[pointer, 2])
                elif x[int(tree[pointer, 0])] > tree[pointer, 1]:
                    pointer += int(tree[pointer, 3])

            pred_y = np.append(pred_y, tree[pointer, 1])

        return pred_y

    def build_tree(self, data):
        if data.shape[0] == 1:
            self.node_count += 1 # adding a leaf node
            return np.array([-1.0, data[0,-1], np.nan, np.nan])
        elif np.min(data[:,-1]) == np.max(data[:,-1]):
            self.node_count += 1 # adding a leaf node
            return np.array([-1.0, data[0,-1], np.nan, np.nan])
        elif data.shape[0] <= self.leaf_size:
            self.node_count += 1 # adding a leaf node
            return np.array([-1.0, np.median(data[:,-1]), np.nan, np.nan]) # aggregate rows using median
        else:
            feature_index = self.get_best_feature(data[:,0:-1], data[:,-1])
            split_val = np.median(data[:, feature_index])

            if self.verbose:
                print("Splitting across feature: ", feature_index)
                print("Splitting value: ", split_val)
                print("\n")

            # check if all elements are <= split value, to avoid recursion depth limit error
            if np.all(data[:,feature_index] <= split_val):
                split_val = np.mean(data[:, feature_index]) # take mean instead of median

            self.node_count += 1 # adding a decision node

            left_tree_data = data[data[:,feature_index] <= split_val]
            left_tree = self.build_tree(left_tree_data)
            right_tree_data = data[data[:,feature_index] > split_val]
            right_tree = self.build_tree(right_tree_data)

            if len(left_tree.shape) == 1:
                root = np.array([feature_index, split_val, 1, 1 + 1]) # left_tree has just one row
            else:
                root = np.array([feature_index, split_val, 1, left_tree[:,0].shape[0] + 1])

            return np.row_stack((root, left_tree, right_tree)) # https://numpy.org/doc/1.25/reference/generated/numpy.row_stack.html

    def get_best_feature(self, data_x, data_y):
        """
        Returns the feature with highest correlation
        """
        corr = np.array([], dtype='float')
        for col in data_x.T:
            cc = np.corrcoef(col, data_y)
            corr = np.append(corr, cc[0, 1])

        # if all values of the feature with high correlation is same, then choose the second best option
        while np.min(data_x[:, np.nanargmax(np.abs(corr))]) == np.max(data_x[:, np.nanargmax(np.abs(corr))]):
            corr[np.nanargmax(np.abs(corr))] = 0

        return np.nanargmax(np.abs(corr))

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
