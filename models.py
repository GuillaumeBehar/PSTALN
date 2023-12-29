import torch
from UDTagSet import UDTagSet

class Hp:
    """
    Hyper-parameters of the tagger
    """
    hidden_size = 64
    embedding_dim = 50
    nb_iter = 3
    dicoUpos = UDTagSet()
    output_dim = len(dicoUpos)
    drop_prob = 0.2
    n_layers = 1


class GRUNet(torch.nn.Module):
    """
    POS tagger model using recurrent neural net architecture
    """

    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, n_layers=1, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.emb = torch.nn.Embedding(vocabSize, input_dim)
        self.gru = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                                dropout=drop_prob if n_layers > 1 else 0)
        self.do = torch.nn.Dropout(p=drop_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(self.do(out)))
        return out, h