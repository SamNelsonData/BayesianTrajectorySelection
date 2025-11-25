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

from sciwrld import SciWrld, Cloud, RewardNet

import sys
import time

seed(42)

temp = SciWrld(
    size=(12,12),
    starting_seeds=8,
    rocks = 25
    )

steps = 10
temp.step(40)
for step in range(steps):
    sys.stdout.write(str(temp))
    sys.stdout.flush()
    sys.stdout.write("\033[12A")
    temp.step()
    time.sleep(.5)
sys.stdout.write("Done")