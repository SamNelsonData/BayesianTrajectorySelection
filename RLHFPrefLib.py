
from torch import exp, sigmoid
from collections.abc import Iterable

'''
Estimates which action is preferred for a given state
@param seg         : the agent's state
@param actions     : action-pair to compare
@param reward_func : reward function used to estimate preference

@return            : probability [0,1] of one action being
                     preferred over the other
'''
def pref_estimation(seg, reward_func):
    r1 = reward_func(seg[0])
    r2 = reward_func(seg[1])

    return sigmoid(r1 - r2)
