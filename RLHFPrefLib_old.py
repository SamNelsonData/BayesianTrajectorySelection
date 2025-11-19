
from torch import exp, log
from collections.abc import Iterable

'''
Estimates which action is preferred for a given state
@param seg         : the agent's state
@param actions     : action-pair to compare
@param reward_func : reward function used to estimate preference

@return            : probability [0,1] of one action being
                     preferred over the other
'''
def pref_estimation(seg, actions, reward_func):
    r1 = reward_func(seg, actions[0])
    e1 = exp(reward_func(seg, actions[0]))
    e2 = exp(reward_func(seg, actions[1]))

    return e1 / (e1 + e2)

def __CrossEntLoss(x, y):
    return -1*((1 - y)*log(x) + y*log(1-x))

def CrossEntropyLoss(states, actions, preference, reward_func):
    if not isinstance(preference, Iterable):
        prob = pref_estimation(states, actions, reward_func)
        return __CrossEntLoss(prob, preference)
    loss = 0
    for state, ap, pref in zip(states, actions, preference):
        prob = pref_estimation(state, ap, reward_func)
        loss += __CrossEntLoss(prob, pref)

    return loss
