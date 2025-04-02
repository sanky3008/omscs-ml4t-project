""""""  		  	   		 	 	 			  		 			     			  	 
"""Assess a betting strategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
import numpy as np
import matplotlib.pyplot as plt
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def author():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return "sphadnis9"  # replace tb34 with your Georgia Tech username.
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def gtid():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    return 904081199  # replace with your GT ID number
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			     			  	 
    :type win_prob: float  		  	   		 	 	 			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	 	 			  		 			     			  	 
    :rtype: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    result = False  		  	   		 	 	 			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		 	 	 			  		 			     			  	 
        result = True  		  	   		 	 	 			  		 			     			  	 
    return result  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Method to test your code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    win_prob = 0.4737  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			     			  	 
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    episodes = 1000
    spins = 1001

    """
    EXPERIMENT 1
    """

    # initialise betting record
    exp1_fig1 = np.zeros((10, spins))
    exp1_fig2 = np.zeros((episodes, spins))

    bet_amount = 1 #initialise starting bet amount

    results = open("p1_results.txt", "w")

    # Run simulation for figure 1
    exp1_fig1 = strategy1(exp1_fig1, win_prob)
    plot_winnings(exp1_fig1, "Exp 1A - Episode Winnings", spins)
    results.write("EXPERIMENT 1A\n\n")
    results.write(f"\nExp 1A Expected Value after 1000 spins: {exp1_fig1[:, -1].mean(axis=0)}")
    results.write(f"\nExp 1A Median Value after 1000 spins: {np.median(exp1_fig1[:, -1], axis=0)}")

    # Run simulation for figure 2
    exp1_fig2 = strategy1(exp1_fig2, win_prob)
    plot_winnings(exp1_fig2, "Exp 1B - Episode Winnings", spins)
    plot_mean(exp1_fig2, "Exp 1B - Mean Winning of each Spin", spins)
    plot_median(exp1_fig2, "Exp 1B - Median Winning of each Spin", spins)

    # Terminal Output
    results.write("\n\nEXPERIMENT 1B\n\n")
    results.write(f"Results after 1000 spins:\n{exp1_fig2[:,-1]}")
    results.write(f"\n\nDistinct Results after 1000 spins:\n{np.unique(exp1_fig2[:,-1],return_counts=True)}")
    results.write(f"\n\nExp 1B Expected Value after 1000 spins:\n{exp1_fig2[:,-1].mean(axis=0)}")
    results.write(f"\n\nExp 1B Median Value after 1000 spins:\n{np.median(exp1_fig2[:, -1], axis=0)}")

    """
    EXPERIMENT 2
    """

    # initialise betting record
    exp2_fig = np.zeros((episodes, spins))

    bet_amount = 1  # initialise starting bet amount

    # Run simulation
    exp2_fig = strategy2(exp2_fig, win_prob)
    plot_winnings(exp2_fig, "Exp 2 - Episode Winnings", spins)
    plot_mean(exp2_fig, "Exp 2 - Mean Winning of each Spin", spins)
    plot_median(exp2_fig, "Exp 2 - Median Winning of each Spin", spins)

    # Terminal Output
    results.write("\n\nEXPERIMENT 2\n\n")
    results.write(f"Results after 1000 spins:\n{exp2_fig[:,-1]}")
    results.write(f"\n\nDistinct Results after 1000 spins:\n{np.unique(exp2_fig[:,-1], return_counts=True)}")
    results.write(f"\n\nExperiment 2 Expected Value after 1000 spins:\n{exp2_fig[:,-1].mean(axis=0)}")
    results.write(f"\n\nExperiment 2 Median Value after 1000 spins:\n{np.median(exp2_fig[:, -1], axis=0)}")

    results.close()

def strategy1(bet_record, win_prob):
    for episode in bet_record:
        bet_amount = 1
        for i in range(1, len(episode)):
            episode_winning = episode[i-1]
            if episode_winning < 80:
                won = get_spin_result(win_prob)
                if won:
                    episode[i] = episode_winning + bet_amount
                    bet_amount = 1
                else:
                    episode[i] = episode_winning - bet_amount
                    bet_amount = bet_amount*2
            else:
                episode[i] = episode_winning

    return bet_record

def strategy2(bet_record, win_prob):
    for episode in bet_record:
        bet_amount = 1
        for i in range(1, len(episode)):
            episode_winning = episode[i-1]
            if 80 > episode_winning > -256:
                won = get_spin_result(win_prob)
                if won:
                    episode[i] = episode_winning + bet_amount
                    bet_amount = 1
                else:
                    episode[i] = episode_winning - bet_amount
                    bet_amount = min(bet_amount*2, 256 + episode[i]) #user has only $256 in wallet
            else:
                episode[i] = episode_winning

    return bet_record

def plot_winnings(bet_record, title, spins):
    x_axis = np.arange(0, spins, 1)
    for episode in bet_record:
        plt.plot(x_axis, episode)
    plt.xlim(0, 300)
    plt.xlabel("Spins")

    plt.ylim(-256, 100)
    plt.ylabel("Winning")

    plt.title(title)
    plt.savefig("images/" + title + ".png")
    plt.close()

def plot_mean(bet_record, title, spins):
    x_axis = np.arange(0, spins, 1)
    mean = bet_record.mean(axis=0)
    stddev = bet_record.std(axis=0)
    plt.plot(x_axis, mean)
    plt.plot(x_axis, mean - stddev)
    plt.plot(x_axis, mean + stddev)
    plt.xlim(0, 300)
    plt.xlabel("Spins")

    plt.ylim(-256, 100)
    plt.ylabel("Mean Winning")
    plt.title(title)
    plt.savefig("images/" + title + ".png")
    plt.close()

def plot_median(bet_record, title, spins):
    x_axis = np.arange(0, spins, 1)
    median = np.median(bet_record, axis=0)
    stddev = bet_record.std(axis=0)
    plt.plot(x_axis, median)
    plt.plot(x_axis, median - stddev)
    plt.plot(x_axis, median + stddev)
    plt.xlim(0, 300)
    plt.xlabel("Spins")

    plt.ylim(-256, 100)
    plt.ylabel("Median Winning")
    plt.title(title)
    plt.savefig("images/" + title + ".png")
    plt.close()
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
