# Bayesian Trajectory Selection
This project aims to improve trajectory selection for human labeling in RLHF by modeling the distribution of the reward function. Trajectories with higher variance in reward are better candidates for labeling.

# Files
## PrefRL.ipynb
This file contains an implementation of RLHF for adding two integers. Preferences are automatically calculated saving time.

## sciwrld.py
This file contains a test environment for our RL agent. An example of how to use this test environment can be found in `temp.py`.

## temp.py
Used for testing code, currently contains an example of how to use `sciwrld.py`
