import sys
print("Python version:", sys.version)
# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.distributions as dis
from tools import *

policy = nn.Sequential(
    nn.Linear(13, 6)
)
policy.load_state_dict(torch.load("policy", weights_only = True))

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.

    state = torch.tensor(get_state(obs), dtype = torch.float)
    probs = nn.functional.softmax(policy(state), dim = 0)
    m = dis.Categorical(probs).sample()

    return m.item()
