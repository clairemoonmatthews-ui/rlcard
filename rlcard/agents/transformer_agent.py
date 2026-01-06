import torch
import torch.nn as nn

class TransformerAgent(nn.Module):
    def __init__(self, vocab_size, actions, max_seq_len, embedding_dimension=32, n_attention_head=4, layers=2, dim_feed_forward=256):
        super().__init__()
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

    def forward(self, obs, legal_mask=None):
        batch_size, seq_len = obs.shape
        x = self.embedding(obs)
        positions = torch.arange(seq_len, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.pos_embedding(positions)
        x += pos_embedding
        x = self.stack(x)
        x = x[:,-1,:]
        policy = self.policy(x)
        if legal_mask is not None:
            policy = policy.masked_fill(~legal_mask, float('-inf'))
        return policy

    def save(self, file_name):
        data = dict(vocab_size=self.embedding.num_embeddings, actions=self.policy.out_features, max_seq_len=self.pos_embedding.num_embeddings, embedding_dimension=self.embedding.embedding_dim, n_attention_head=self.stack.layers[0].self_attn.num_heads, layers=len(self.stack.layers), dim_feed_forward=self.stack.layers[0].linear1.out_features, state=self.state_dict())
        torch.save(data, file_name)

    @classmethod
    def load(cls, file_name, device="cpu", train=True):
        data = torch.load(file_name, map_location=device)
        agent = cls(vocab_size=data['vocab_size'], actions=data['actions'], max_seq_len=data['max_seq_len'], embedding_dimension=data['embedding_dimension'], n_attention_head=data['n_attention_head'], layers=data['layers'], dim_feed_forward=data['dim_feed_forward'])
        agent.load_state_dict(data['state'])
        agent.to(device)
        agent.train(train)
        return agent
