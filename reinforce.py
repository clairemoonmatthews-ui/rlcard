#!/bin/env python3
from datetime import datetime, timezone
import logging
import os

import numpy as np
import torch

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.sergeantmajor_agent import HeuristicAgent
from rlcard.agents.transformer_agent import TransformerAgent
from rlcard.envs.env import Env

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def resolve_input_path(path):
    """
    If path is a directory, return the lexically last (most recent) file in it.
    Otherwise, return the path as-is.
    """
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        if not files:
            raise ValueError(f"Directory {path} contains no files")
        selected_file = files[-1]  # Lexically last
        resolved_path = os.path.join(path, selected_file)
        logger.info(f"Input is directory, selected most recent file: {resolved_path}")
        return resolved_path
    return path


def resolve_output_path(path):
    """
    If path is a directory, generate a filename based on ISO date-seconds-UTC.
    Otherwise, return the path as-is.
    """
    if os.path.isdir(path):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"model_{timestamp}.pt"
        resolved_path = os.path.join(path, filename)
        logger.info(f"Output is directory, using generated filename: {resolved_path}")
        return resolved_path
    return path


def make_agent(file_name):
    input_path = resolve_input_path(file_name)
    agent = TransformerAgent.load(input_path)
    return agent

def make_env() -> "Env":
    env = rlcard.make('sergeant-major')
    return env

def save_agent(file_name, agent, epoch=None):
    output_path = resolve_output_path(file_name)
    # If saving per epoch, insert epoch number into filename
    if epoch is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_epoch{epoch}{ext}"
    agent.save(output_path)
    logger.info(f"Model saved to: {output_path}")
    return output_path
    

def play_eachother(agent):
    env = make_env() 
    agents = [agent] * 3
    env.set_agents(agents)
    trajectories, payoffs = env.run(is_training=True)
    # TODO: Think about ties
    win = [1 if payoff == max(payoffs) else -1 for payoff in payoffs] 
    data = []
    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        for j in range(0, len(trajectory) - 1, 2):
            state = trajectory[j]
            action = trajectory[j + 1]
            data.append((state, action, win[i]))
    return data

def training(agent, n=10000, lr=1e-6):
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    for i in range(n):
        total_loss = 0
        data = play_eachother(agent)
        for state, action, win in data:
            obs = torch.tensor(state['obs'], dtype=torch.long).unsqueeze(0).to(agent.device)
            logits = agent(obs)
            log_probs = torch.log_softmax(logits, dim=-1)
            total_loss -= log_probs[0][action] * win
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if n%100 == 0:
            run_tournaments(agent, epoch=i)
        if n%100 == 0:
            save_agent("models", agent, epoch=i)


def run_tournament(agent, heuristic=False):
    n_games = 1000
    wins = 0
    r_agent = RandomAgent(52)
    h_agent = HeuristicAgent()
    opponent = h_agent if heuristic else r_agent
    for _ in range(n_games):
        position = np.random.randint(3)
        env = make_env()
        agents = [opponent] * 3
        agents[position] = agent
        env.set_agents(agents)
        trajectories, payoffs = env.run(is_training=False)
        win = payoffs[position] == max(payoffs) # TODO: worry about ties
        # print(payoffs, position, win)
        if win:
            wins += 1
    win_rate = wins / n_games
    return win_rate

def run_tournaments(agent, epoch=None):
    r_win_rate = run_tournament(agent, False)
    h_win_rate = run_tournament(agent, True)
    print(f"Win rate: {r_win_rate:%} (Random), {h_win_rate:%} (Heuristic), Epoch={epoch}")


input_file_name = "models/model_20260107T042047Z_epoch24.pt"
agent = make_agent(input_file_name)
training(agent)
save_agent("models", agent)
run_tournaments(agent)