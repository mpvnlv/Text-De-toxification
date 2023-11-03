import torch.nn as nn
import torch.nn.functional as F


class SimpleParaphraseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, sequence_length):
        super(SimpleParaphraseModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, reference, length_diff):
        embeds = self.embeddings(reference)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
