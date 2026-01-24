from contextlib import contextmanager
from typing import Dict, Tuple
import torch
import torch.nn as nn

class TransformerAgent(nn.Module):
    def __init__(self, vocab_size, actions, max_seq_len, embedding_dimension=32, n_attention_head=4, layers=2, nplayers = 3,  dim_feed_forward=256, device='cpu'):
        super().__init__()
        self.use_raw = False  # RLCard convention
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dimension)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=n_attention_head,
            dim_feedforward=dim_feed_forward,
            batch_first=True
        )
        self.stack = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=layers)
        self.policy = nn.Linear(in_features=embedding_dimension, out_features=actions) 
        self.value = nn.Linear(in_features=embedding_dimension, out_features= nplayers) 
        self.actions = actions
        self.device = device

    def forward(self, obs, legal_mask=None):
        assert str(obs.device) == self.device, f"obs={obs.device}, self={self.device}"
        batch_size, seq_len = obs.shape
        x = self.embedding(obs)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.pos_embedding(positions)
        x += pos_embedding
        x = self.stack(x)
        x = x[:,-1,:]
        policy = self.policy(x)
        if legal_mask is not None:
            policy = policy.masked_fill(~legal_mask, float('-inf'))
        value = self.value(x)
        value = torch.softmax(value, dim=-1) # we are storing a value for each player and we are making them sum to 1
        return policy, value

    def save(self, file_name):
        data = dict(vocab_size=self.embedding.num_embeddings, actions=self.policy.out_features, max_seq_len=self.pos_embedding.num_embeddings, embedding_dimension=self.embedding.embedding_dim, n_attention_head=self.stack.layers[0].self_attn.num_heads, layers=len(self.stack.layers), dim_feed_forward=self.stack.layers[0].linear1.out_features, state=self.state_dict())
        torch.save(data, file_name)


    def step(self, state:Dict) -> int:
        """
        Called during gameplay to select an action.
        
        Args:
            state: Dict with 'obs' (token sequence) and 'legal_actions' (list of valid card indices)
        
        Returns:
            action: Integer in range [0, 51] representing card to play
        """
        legal_actions = state['legal_actions']
        with self.training_mode(False):
            with torch.no_grad():
                obs = torch.tensor(state['obs'], dtype=torch.long).unsqueeze(0).to(self.device)
                legal_actions = torch.tensor(list(legal_actions.keys()), dtype=torch.long, device=self.device)
                logits, _ = self(obs)
                mask = torch.ones(self.actions, dtype=torch.bool, device=self.device)
                mask[legal_actions] = False
                logits = logits.masked_fill(mask, float('-inf'))
                action = logits.argmax(dim = 1).item()
        return action
    
    def get_value(self, state:Dict) -> float:
        """
        Called during gameplay to get the value.
        
        Args:
            state: Dict with 'obs' (token sequence) and 'legal_actions' (list of valid card indices)
        
        Returns:
            estimated value
        """
        with self.training_mode(False):
            with torch.no_grad():
                obs = torch.tensor(state['obs'], dtype=torch.long).unsqueeze(0).to(self.device)
                _, value = self(obs)
        return value.tolist()
    
    @contextmanager 
    def training_mode(self, mode=bool):
        old_mode = self.training
        try: 
            self.train(mode)
            yield
        finally:
            self.train(old_mode)

    
    def eval_step(self, state:Dict) -> Tuple[int, Dict]:
        """
        Called during evaluation - returns action and info dict.
        
        Args:
            state: Same as step()
        
        Returns:
            action: Selected action
            info: Dict with any debugging/analysis info (can be empty)
        """
        action = self.step(state)
        return action, {}

    @classmethod
    def load(cls, file_name, device="cpu", train=True):
        data = torch.load(file_name, map_location=device)
        agent = cls(vocab_size=data['vocab_size'], actions=data['actions'], max_seq_len=data['max_seq_len'], embedding_dimension=data['embedding_dimension'], n_attention_head=data['n_attention_head'], layers=data['layers'], dim_feed_forward=data['dim_feed_forward'])
        agent.load_state_dict(data['state'])
        agent.to(device)
        agent.train(train)
        return agent
