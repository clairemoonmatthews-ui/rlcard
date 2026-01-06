#!/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import torch.nn as nn
import torch
import logging

import rlcard
from rlcard.agents.transformer_agent import TransformerAgent
from rlcard.envs.env import Env
from rlcard.envs.sergeantmajor import SergeantMajorEnv

logging.basicConfig(level=logging.INFO)
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


def load_batches(file_name, max_batch_size):
    buckets = defaultdict(list)
    with open(file_name) as f:
        for line in f:
            d = json.loads(line)
            obs = d['obs']
            actions = d['action']
            buckets[len(obs)].append((obs, actions))
    logger.info(f"Read {sum(len(x) for x in buckets.values())} records from {file_name}")
    batches = []
    for length in sorted(buckets.keys()):
        records = buckets[length]
        for i in range(0,len(records), max_batch_size):
            r = records[i:i+max_batch_size]
            obs = torch.stack([torch.tensor(rec[0], dtype=torch.long) for rec in r])
            actions = torch.tensor([rec[1] for rec in r], dtype=torch.long)
            batches.append((obs, actions))
    logger.info(f"Generated {len(batches)} batches from {len(buckets)} lengths")
    return batches


def make_env(args) -> "Env":
    env = rlcard.make('sergeant-major')
    return env


def make_agent(args, env:SergeantMajorEnv):
    if args.input:
        input_path = resolve_input_path(args.input)
        agent = TransformerAgent.load(input_path)
    else:
        agent = TransformerAgent(vocab_size=env.vocab_size, actions=env.actions, max_seq_len=env.max_state_length)
    return agent


def save_agent(args, agent, epoch=None):
    output_path = resolve_output_path(args.output)
    # If saving per epoch, insert epoch number into filename
    if epoch is not None:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_epoch{epoch}{ext}"
    agent.save(output_path)
    logger.info(f"Model saved to: {output_path}")
    return output_path
    

def train(args):
    batches = load_batches(args.training_data, args.max_batch_size)
    env = make_env(args)
    agent = make_agent(args, env)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    for epoch in range(args.number_epochs):
        total_loss = 0
        total_n = 0
        total_correct = 0
        for obs, actions in batches:
            optimizer.zero_grad()
            predicted_actions = agent(obs)
            loss = loss_func(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_n += len(actions)
            greedy_predictions = predicted_actions.argmax(dim=1)
            total_correct += (greedy_predictions==actions).sum().item()
        print(f"epoch number = {epoch}, average loss = {total_loss/total_n}, accuracy = {total_correct/total_n}")
        if args.save_per_epoch:
            save_agent(args, agent, epoch=epoch)
    # Always save final model (without epoch suffix if not save_per_epoch)
    if not args.save_per_epoch:
        save_agent(args, agent)

def parse_args():
    def str2bool(v):
        return v.lower() in {'true', 't', '1', 'yes', 'y'}

    parser = ArgumentParser()
    parser.add_argument('--training-data', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--number-epochs', type=int, default=1)
    parser.add_argument('--max-batch-size', type=int, default=512)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--save-per-epoch', action='store_true', 
                        help='Save model after each epoch with epoch number in filename')
    return parser.parse_args()

args = parse_args()
train(args)


