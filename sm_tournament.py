#!/bin/env python
import numpy as np
import rlcard
from rlcard.agents import random_agent
from rlcard.agents.mcts_agent import MCTSAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sergeantmajor_agent import HeuristicAgent
from rlcard.agents.transformer_agent import TransformerAgent
from rlcard.envs.env import Env
from rlcard.games.sergeantmajor.game import SergeantMajorGame
from rlcard.utils.utils import tournament

import logging
from tqdm import tqdm 

logging.basicConfig(level=logging.WARNING)

# logging.basicConfig(level=logging.INFO)
competitor = "mcts"
opponent = "heuristic"  # random, heuristic, model
model = "models/model_20260127T020706Z.pt"
game_class = SergeantMajorGame

def make_competitor():
    match competitor:
        case "transformer":
            return TransformerAgent.load(model)
        case "mcts":
            transformer_agent = TransformerAgent.load(model)
            return MCTSAgent(agent=transformer_agent, game_class=game_class)
        case _:
            assert False, f"Unknown competitor {competitor}"

def make_opponent():
    match opponent:
        case "random":
            return RandomAgent(num_actions=env.num_actions)
        case "heuristic":
            return HeuristicAgent()
        case "transformer":
            return TransformerAgent.load(model)
        case _:
            assert False, f"Unknown opponent type: {opponent}"



def make_env() -> "Env":
    env = rlcard.make('sergeant-major')
    return env


def set_agents(env: "Env", position=0):
    agent = make_competitor()
    o_agent = make_opponent()

    agents = [agent if i == position else o_agent 
              for i in range(env.num_players)]
    env.set_agents(agents)


n_games = 1000
wins = 0
for _ in tqdm(range(n_games), desc=f"{competitor} vs {opponent} ({n_games})"):
    position = np.random.randint(3)
    env = make_env()
    set_agents(env, position)
    trajectories, payoffs = env.run(is_training=False)
    win = payoffs[position] == max(payoffs)  # TODO: worry about ties
    # print(payoffs, position, win)
    if win:
        wins += 1
win_rate = wins / n_games
print(f"Competitor={competitor}, Model={model}, Opponent={opponent}, Win rate: {win_rate:%}")
