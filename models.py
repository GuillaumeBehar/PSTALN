import torch
import torch.nn as nn
import torch.optim as optim
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


class GRUNet(nn.Module):
    """
    POS tagger model using recurrent neural net architecture
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 vocabSize: int,
                 n_layers: int = 1,
                 drop_prob=0
                 ):
        super(GRUNet, self).__init__()
        self.emb = nn.Embedding(vocabSize, input_dim)
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=True,
                          dropout=drop_prob if n_layers > 1 else 0)
        self.do = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(self.do(out)))
        return out

class BiGRUNet(nn.Module):
    """
    POS tagger model using recurrent neural net architecture
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 vocabSize: int,
                 n_layers: int = 1,
                 drop_prob: int = 0
                 ):
        super(BiGRUNet, self).__init__()
        self.emb = nn.Embedding(vocabSize, input_dim)
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          n_layers,
                          batch_first=True,
                          dropout=drop_prob if n_layers > 1 else 0,
                          bidirectional=True)
        self.do = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(self.emb(x))
        out = self.fc(self.relu(self.do(out)))
        return out


class BiLSTMPOSTagger(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 pad_idx: int,
                 n_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            bidirectional = True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions


class CharEmbedding(nn.Module):
    """
    Dense character embedding.

    Parameters
    ----------
    n_chars : int
        The number of characters in the vocabularly, i.e. the input size.

    embedding_size : int
        The dimension of the embedding.

    dropout : float, optional (default: 0.)
        The dropout probability.

    padding_idx : int, optional (default: 0)
        The id of the character using for padding.

    """

    def __init__(self,
                 n_chars: int,
                 embedding_size: int,
                 dropout: float = 0.,
                 padding_idx: int = 0) -> None:
        super(CharEmbedding, self).__init__()
        self.embedding = \
            nn.Embedding(n_chars, embedding_size, padding_idx=padding_idx)

        # Dropout applied to embeddings.
        self.embedding_dropout = \
            nn.Dropout(p=dropout) if dropout else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        #print(inputs)
        inputs_emb = self.embedding(inputs)
        # inputs_emb: ``[sent_length x max_word_length x embedding_size]``
        output = self.embedding_dropout(inputs_emb)
        return output

class CNN_GRU(nn.Module):
    """
    Based on Ma & Hovy, 2016
    """

    def __init__(self,
                 char_vocab_size: int,
                 char_embedding_dim: int,
                 n_filters: int,
                 vocab_size: int,
                 word_embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 1,
                 drop_out: float = 0.,
                 char_drop_out: float = 0.1,
                 padding: int = 2):
        super(CNN_GRU,self).__init__()
        self.char_embedding = CharEmbedding(char_vocab_size,
                                            char_embedding_dim,
                                            dropout=char_drop_out,
                                            padding_idx=padding)
        self.cnn = \
            nn.Conv1d(char_embedding_dim, n_filters, kernel_size=3, padding=padding)
        self.cnn_output_size = n_filters
        self.word_emb = nn.Embedding(vocab_size, word_embedding_dim)
        self.gru = nn.GRU(
            word_embedding_dim + n_filters,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_out if n_layers > 1 else 0)
        self.do = nn.Dropout(p=drop_out)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor):
        '''
        inputs is a tensor with shape :sentence_len * [word indice, char indices]
        '''
        char_inputs = inputs[:,1:]
        #print(f"dimension de char_inputs: {char_inputs.shape}")
        inputs_emb = self.char_embedding(char_inputs).permute(0, 2, 1)
        #print(f"dimension de inputs_emb: {inputs_emb.shape}")
        # inputs_emb: ``[sent_length x embedding_size x max_word_length ]``
        output_cnn = self.cnn(inputs_emb)
        #print(f"dimension de l'output cnn: {output_cnn.shape}")
        # output: ``[sent_length x filters x out_length]``
        output_cnn, _ = torch.max(output_cnn, 2)
        #print(f"dimension de l'output cnn: {output_cnn.shape}")
        # output: ``[sent_length x filters]``
        word_emb = self.do(self.word_emb(inputs[:,0]))
        #print(f"dimension des embeddings: {word_emb.shape}")
        input = torch.cat((word_emb,output_cnn), dim=1)
        #print(f"dimension de l'input: {input.shape}")
        # input: ``[sent_len * (word_emb_dim + filters)]
        out, h = self.gru(input)
        out = self.fc(self.relu(self.do(out)))
        return out, h