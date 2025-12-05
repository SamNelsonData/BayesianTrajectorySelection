import numpy as np
from numpy import where, array, zeros
from numpy.random import choice, uniform, normal, seed, binomial

from scipy.optimize import minimize

import jax
import jax.numpy as jnp

import torch
import torch.nn as nn

from torch import exp

from numpy.random import binomial, seed

from RLHFPrefLib import pref_estimation

from sciwrld import SciWrld, Cloud, RewardNet, AgentA2C

import sys
import time

seed(43)

temp = SciWrld(
    size=(15,15),
    starting_seeds=8,
    rocks = 25
    )

temp.add_agent(AgentA2C)

steps = 40
#temp.step(41)
for step in range(steps):
    sys.stdout.write(str(temp))
    sys.stdout.flush()
    sys.stdout.write("\033[15A")
    temp.step()
    time.sleep(.2)
sys.stdout.write("Done\n")