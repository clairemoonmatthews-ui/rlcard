#!/bin/env python
import numpy as np
import rlcard
from rlcard.agents import random_agent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sergeantmajor_agent import HeuristicAgent
from rlcard.envs.env import Env
from rlcard.utils.utils import tournament

import logging

# logging.basicConfig(level=logging.INFO)

def make_env() -> "Env":
    env = rlcard.make('sergeant-major')
    return env

def set_agents(env: "Env", position=0):
    random_agent = RandomAgent(num_actions=env.num_actions)
    agent = HeuristicAgent()
    agents = [random_agent, random_agent, random_agent]
    agents[position] = agent
    env.set_agents(agents)

n_games = 1000
wins = 0
for _ in range(n_games):
    position = np.random.randint(3)
    env = make_env()
    set_agents(env, position)
    trajectories, payoffs = env.run(is_training=False)
    win = payoffs[position] == max(payoffs) # TODO: worry about ties
    # print(payoffs, position, win)
    if win:
        wins += 1
win_rate = wins / n_games
print(f"Win rate: {win_rate:%}")

