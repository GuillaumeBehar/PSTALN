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

class BiGRUNet(torch.nn.Module):
    """
    POS tagger model using recurrent neural net architecture
    """

    def __init__(self, input_dim, hidden_dim, output_dim, vocabSize, n_layers=1, drop_prob=0.5):
        super(BiGRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.emb = torch.nn.Embedding(vocabSize, input_dim)
        self.gru = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                                dropout=drop_prob if n_layers > 1 else 0, bidirectional=True)
        self.do = torch.nn.Dropout(p=drop_prob)
        self.fc = torch.nn.Linear(2*hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(self.do(out)))
        return out, h

class FeedForward(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, dropout_rate=0.5):
        super(FeedForward, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.l1 = torch.nn.Linear(embedding_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        e = self.embedding(x)
        h_ = self.l1(e)
        h = self.relu(h_)
        y = self.l2(self.dropout(h))
        return y

class GRU_prtEmb(torch.nn.Module):
    def __init__(self, embedding, input_dim, hidden_dim, output_dim, n_layers=1, drop_prob=0.3):
        super(GRU_prtEmb, self).__init__()
        self.embedding = embedding
        #self.embedding.requires_grad = True
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.gru = torch.nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True,
                                dropout=drop_prob if n_layers > 1 else 0)
        self.do = torch.nn.Dropout(p=drop_prob)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out, h = self.gru(self.embedding(x))
        out = self.fc(self.relu(self.do(out)))
        return out, h

class MLB_RNN(torch.nn.Module):
    """
    Multi-Level Bidirectional Recurrent Neural Network
    Model using Bidirectional LSTM with Flair embeddings for POS tagging added followed by a
    Bidirectional GRU-RNN architecture at character level for Lemmatization.
    """

    def __init__(self,POSmodel,letter_dict_path,embedding_dim, hidden_dim, output_dim, n_layers=1, drop_prob=0.3):
        super(MLB_RNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Pos_model = POSmodel.eval()
        self.letter_dict_path = letter_dict_path
        self.emb = torch.nn.Embedding(output_dim, embedding_dim)
        self.gru = torch.nn.GRU(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_prob if n_layers > 1 else 0)
        self.do = torch.nn.Dropout(p=drop_prob)
        self.fc = torch.nn.Linear(2*hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

        def forward(self, x):
            emb = self.emb(x)
            input = torch.concat((emb,pos))
            out, h = self.gru(input)
            out = self.fc(self.relu(self.do(out)))
            return out, h