""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
import sys  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  		  	   		 	 	 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it


def author(self):
    """
    Returns the author
    """
    return "sphadnis9"
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    results = open("p3_results.txt", "w")

    # if Istanbul.csv is taken, we'll have to clean the data a bit
    is_istanbul = False
    if sys.argv[1] == "Data/Istanbul.csv":
        is_istanbul = True

    data = np.genfromtxt(inf, delimiter=",")

    # ignore header row and date column in Istanbul.csv
    if is_istanbul:
        data = data[1:, 1:]

    # prepare training & testing data
    datasize = data.shape[0]
    cutoff = int(datasize * 0.6)
    permutation = np.random.permutation(data.shape[0])
    col_permutation = np.random.permutation(data.shape[1] - 1)
    train_data = data[permutation[:cutoff], :]

    train_x = train_data[:, col_permutation]
    train_y = train_data[:, -1]
    test_data = data[permutation[cutoff:], :]

    test_x = test_data[:, col_permutation]
    test_y = test_data[:, -1]

    results.write(f"train_x: {test_x.shape}")
    results.write(f"train_y: {test_y.shape}")

    results.write(f"test_x: {test_x.shape}")
    results.write(f"test_y: {test_y.shape}")

    # run experiments 1 & 2
    results.write("\nEXPERIMENT 1 & 2:\n\n")
    dt_rmse_data = np.empty((50, 2))
    bag_rmse_data = np.empty((50, 2))
    dt_corr_data = np.empty((50, 2))
    bag_corr_data = np.empty((50, 2))

    for leaf_size in range(1, 51):
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"verbose": False, "leaf_size": leaf_size}, bags = 20, verbose=False)

        dt_learner.add_evidence(train_x, train_y)
        bag_learner.add_evidence(train_x, train_y)

        results.write("\n\nLEAF SIZE " + str(leaf_size) + ":\n")

        # In sample testing
        dt_pred_y = dt_learner.query(train_x)
        bag_pred_y = bag_learner.query(train_x)

        results.write("\nIn sample testing for leaf size " + str(leaf_size) + ":\n")
        in_dt_rmse = math.sqrt(((train_y - dt_pred_y) ** 2).sum() / train_y.shape[0])
        in_bag_rmse = math.sqrt(((train_y - bag_pred_y) ** 2).sum() / train_y.shape[0])

        results.write("RMSE DTLearner: " + str(in_dt_rmse) + "\n")
        results.write("RMSE BagLearner: " + str(in_bag_rmse) + "\n")

        in_dt_c = np.corrcoef(dt_pred_y, y=train_y)[0,1]
        in_bag_c = np.corrcoef(bag_pred_y, y=train_y)[0,1]

        results.write("Corr DTLearner: " + str(in_dt_c) + "\n")
        results.write("Corr BagLearner: " + str(in_bag_c) + "\n")

        # Out of sample testing
        dt_pred_y = dt_learner.query(test_x)
        bag_pred_y = bag_learner.query(test_x)

        results.write("\nOut sample testing for leaf size " + str(leaf_size) + ":\n")
        out_dt_rmse = math.sqrt(((test_y - dt_pred_y) ** 2).sum() / test_y.shape[0])
        out_bag_rmse = math.sqrt(((test_y - bag_pred_y) ** 2).sum() / test_y.shape[0])

        results.write("RMSE DTLearner: " + str(out_dt_rmse) + "\n")
        results.write("RMSE BagLearner: " + str(out_bag_rmse) + "\n")

        out_dt_c = np.corrcoef(dt_pred_y, y=test_y)[0,1]
        out_bag_c = np.corrcoef(bag_pred_y, y=test_y)[0,1]

        results.write("Corr DTLearner: " + str(out_dt_c) + "\n")
        results.write("Corr BagLearner: " + str(out_bag_c) + "\n")

        dt_rmse_data[leaf_size-1, :] = np.array([in_dt_rmse, out_dt_rmse])
        bag_rmse_data[leaf_size-1, :] = np.array([in_bag_rmse, out_bag_rmse])
        dt_corr_data[leaf_size-1, :] = np.array([in_dt_c, out_dt_c])
        bag_corr_data[leaf_size-1, :] = np.array([in_bag_c, out_bag_c])

    x_axis = np.arange(1, 51)
    plt.plot(x_axis, dt_rmse_data[:, 0], label="DTLearner In Sample RMSE")
    plt.plot(x_axis, dt_rmse_data[:, 1], label="DTLearner Out Sample RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("DT Learner RMSE v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/dt_rmse.png")
    plt.close()

    plt.plot(x_axis, bag_rmse_data[:, 0], label="BagLearner In Sample RMSE")
    plt.plot(x_axis, bag_rmse_data[:, 1], label="BagLearner Out Sample RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("Bag Learner RMSE v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/bag_rmse.png")
    plt.close()

    plt.plot(x_axis, dt_corr_data[:, 0], label="DTLearner In Sample Correlation")
    plt.plot(x_axis, dt_corr_data[:, 1], label="DTLearner Out Sample Correlation")
    plt.xlabel("Leaf Size")
    plt.ylabel("DT Learner Correlation v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/dt_corr.png")
    plt.close()

    plt.plot(x_axis, bag_corr_data[:, 0], label="BagLearner In Sample Correlation")
    plt.plot(x_axis, bag_corr_data[:, 1], label="BagLearner Out Sample Correlation")
    plt.xlabel("Leaf Size")
    plt.ylabel("Bag Learner Correlation v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/bag_corr.png")
    plt.close()


    # run experiment 3
    results.write("\n\nEXPERIMENT 3:\n\n")
    dt_mae_data = np.empty((50, 2))
    rt_mae_data = np.empty((50, 2))
    dt_nodecnt_data = np.empty(50)
    rt_nodecnt_data = np.empty(50)

    for leaf_size in range(1, 51):
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)

        dt_learner.add_evidence(train_x, train_y)
        rt_learner.add_evidence(train_x, train_y)

        results.write("\n\nLEAF SIZE " + str(leaf_size) + ":\n")

        # In sample testing
        dt_pred_y = dt_learner.query(train_x)
        rt_pred_y = rt_learner.query(train_x)

        results.write("\nIn sample testing for leaf size " + str(leaf_size) + ":\n")
        in_dt_mae = np.median(np.abs(dt_pred_y - train_y))
        in_rt_mae = np.median(np.abs(rt_pred_y - train_y))

        results.write("Median Abs Error DTLearner: " + str(in_dt_mae) + "\n")
        results.write("Median Abs Error RTLearner: " + str(in_rt_mae) + "\n")

        # Out of sample testing
        dt_pred_y = dt_learner.query(test_x)
        rt_pred_y = rt_learner.query(test_x)

        results.write("\nOut sample testing for leaf size " + str(leaf_size) + ":\n")
        out_dt_mae = np.median(np.abs(dt_pred_y - test_y))
        out_rt_mae = np.median(np.abs(rt_pred_y - test_y))

        results.write("Median Abs Error DTLearner: " + str(out_dt_mae) + "\n")
        results.write("Median Abs Error RTLearner: " + str(out_rt_mae) + "\n")

        # Get node count
        dt_nodecnt = dt_learner.node_count
        rt_nodecnt = rt_learner.node_count

        results.write("DT Node Count: " + str(dt_nodecnt) + "\n")
        results.write("RT Node Count: " + str(rt_nodecnt) + "\n")

        dt_mae_data[leaf_size - 1, :] = np.array([in_dt_mae, out_dt_mae])
        rt_mae_data[leaf_size - 1, :] = np.array([in_rt_mae, out_rt_mae])
        dt_nodecnt_data[leaf_size - 1] = dt_nodecnt
        rt_nodecnt_data[leaf_size - 1] = rt_nodecnt

    x_axis = np.arange(1, 51)
    plt.plot(x_axis, dt_mae_data[:, 0], label="DTLearner In Sample Median Abs Error")
    plt.plot(x_axis, rt_mae_data[:, 0], label="RTLearner In Sample Median Abs Error")
    plt.xlabel("Leaf Size")
    plt.ylabel("In-sample Median Abs Error v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/exp3_insample_mae.png", bbox_inches='tight')
    plt.close()

    plt.plot(x_axis, dt_mae_data[:, 1], label="DTLearner Out Sample Median Abs Error")
    plt.plot(x_axis, rt_mae_data[:, 1], label="RTLearner Out Sample Median Abs Error")
    plt.xlabel("Leaf Size")
    plt.ylabel("Out-sample Median Abs Error v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/exp3_outsample_mae.png", bbox_inches='tight')
    plt.close()

    plt.plot(x_axis, dt_nodecnt_data, label="DTLearner Node Count")
    plt.plot(x_axis, rt_nodecnt_data, label="RTLearner Node Count")
    plt.xlabel("Leaf Size")
    plt.ylabel("Node Count v/s Leaf Size")
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig("images/exp3_nodecount.png", bbox_inches='tight')
    plt.close()

    results.close()