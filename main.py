import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
import base64, io

from collections import deque, namedtuple

# For visualization


from src.agent import Agent
from src.train import dqn
import src.config as config

env = gym.make('LunarLander-v2')
observation, info = env.reset(seed=42)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = Agent(state_size=8, action_size=4, seed=0, device=device)
scores = dqn(agent, env, device, n_episodes=10)

if config.visualise:
    agent = Agent(state_size=8, action_size=4, seed=0)
    show_video_of_model(agent, 'LunarLander-v2')
    show_video('LunarLander-v2')