#!/bin/env python
import numpy as np
import rlcard
from rlcard.agents import random_agent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sergeantmajor_agent import HeuristicAgent
from rlcard.agents.transformer_agent import TransformerAgent
from rlcard.envs.env import Env
from rlcard.utils.utils import tournament

import logging

# logging.basicConfig(level=logging.INFO)
opponent = "heuristic"  # random, heuristic, model
model = "models/model_20260127T020706Z.pt"


def make_env() -> "Env":
    env = rlcard.make('sergeant-major')
    return env


def set_agents(env: "Env", position=0):
    input_path = model
    agent = TransformerAgent.load(input_path)

    if opponent == "random":
        o_agent = RandomAgent(num_actions=env.num_actions)
    elif opponent == "heuristic":
        o_agent = HeuristicAgent()
    elif opponent == "model":
        o_agent = agent
    else:
        assert False, f"Unknown opponent type: {opponent}"

    input_path = model
    agent = TransformerAgent.load(input_path)
    agents = [agent if i ==
              position else o_agent for i in range(env.num_players)]
    env.set_agents(agents)


n_games = 1000
wins = 0
for _ in range(n_games):
    position = np.random.randint(3)
    env = make_env()
    set_agents(env, position)
    trajectories, payoffs = env.run(is_training=False)
    win = payoffs[position] == max(payoffs)  # TODO: worry about ties
    # print(payoffs, position, win)
    if win:
        wins += 1
win_rate = wins / n_games
print(f"Model={model}, Opponent={opponent}, Win rate: {win_rate:%}")
