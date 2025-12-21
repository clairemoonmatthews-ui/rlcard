from argparse import ArgumentParser
from collections import defaultdict
import json
import torch.nn as nn
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logger.getlogger(__name__)

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


def make_agent(args):
    pass

def save_agent(args):
    pass

def train(args):
    batches = load_batches(args.training_data, args.max_batch_size)
    agent = make_agent(args)
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
    save_agent(args)

def parse_args():
    def str2bool(v):
        return v.lower() in {'true', 't', '1', 'yes', 'y'}

    parser = ArgumentParser()
    parser.add_argument('--training-data', type=str)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--number-epochs', type=int, default=1)
    parser.add_argument('--max-batch-size', type=int, default=512)
    return parser.parse_args()

args = parse_args()
train(args)
