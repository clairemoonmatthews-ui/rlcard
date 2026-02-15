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
short_model = model.split("/")[-1].split(".")[0]
mcts_num_simulations = 10
game_class = SergeantMajorGame
competitor_name = None
opponent_name = None
competitor_agent = None
opponent_agent = None


def make_agent(key: str):
    match key:
        case "random":
            return RandomAgent(num_actions=env.num_actions), f"random"
        case "heuristic":
            return HeuristicAgent(), f"heuristic"
        case "transformer":
            return TransformerAgent.load(model), f"transformer({short_model})"
        case "mcts":
            transformer_agent = TransformerAgent.load(model)
            name = f"mcts({short_model}, {mcts_num_simulations})"
            return MCTSAgent(agent=transformer_agent, game_class=game_class, num_simulations=mcts_num_simulations), name
        case _:
            assert False, f"Unknown competitor {competitor}"


def make_env() -> "Env":
    env = rlcard.make('sergeant-major')
    return env


def set_agents(env: "Env", competitor_agent, opponent_agent, position=0):
    agents = [competitor_agent if i == position else opponent_agent
              for i in range(env.num_players)]
    env.set_agents(agents)


# Create agents once before the tournament
competitor_agent, competitor_name = make_agent(competitor)
opponent_agent, opponent_name = make_agent(opponent)

n_games = 100
wins = 0
bar = tqdm(range(n_games), desc=f"{competitor_name} vs {opponent_name}")
for _ in bar:
    position = np.random.randint(3)
    env = make_env()
    set_agents(env, competitor_agent, opponent_agent, position)
    trajectories, payoffs = env.run(is_training=False)
    # Clear MCTS cache between games if applicable
    if hasattr(competitor_agent, 'reset_cache'):
        competitor_agent.reset_cache()
    if hasattr(opponent_agent, 'reset_cache'):
        opponent_agent.reset_cache()
    win = payoffs[position] == max(payoffs)  # TODO: worry about ties
    # print(payoffs, position, win)
    if win:
        wins += 1
    bar.set_postfix({"win_rate": f"{wins / (bar.n + 1):.2%}"})
win_rate = wins / n_games
print(
    f"Competitor={competitor_name}, Opponent={opponent_name}, Win rate: {win_rate:.2%}")

# Print cache statistics if available
if hasattr(competitor_agent, 'get_cache_stats'):
    stats = competitor_agent.get_cache_stats()
    print(f"Competitor cache stats: {stats['hits']} hits, {stats['misses']} misses, "
          f"{stats['hit_rate']:.1%} hit rate, {stats['cache_size']} unique states")
if hasattr(opponent_agent, 'get_cache_stats'):
    stats = opponent_agent.get_cache_stats()
    print(f"Opponent cache stats: {stats['hits']} hits, {stats['misses']} misses, "
          f"{stats['hit_rate']:.1%} hit rate, {stats['cache_size']} unique states")
