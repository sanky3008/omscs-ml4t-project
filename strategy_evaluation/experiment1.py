import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

import TheoreticallyOptimalStrategy as TOS
from strategy_evaluation.marketsimcode import compute_portvals, compute_stats
import indicators
from strategy_evaluation.ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
import random as rand
import time


def author():
    return "sphadnis9"

def study_group():
    return "sphadnis9"