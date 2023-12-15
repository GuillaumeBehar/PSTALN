
import torch
import json

########################################################################

try :
    device = torch.device("cuda")
except :    
    device = torch.device("cpu")

########################################################################

class GRUParameters():
    """
    Parameters of the GRUNet
    """
    def __init__(self, output_dim:int, hidden_dim:int=64, embedding_dim:int=50, nb_iter:int=1, drop_prob:float=0.2, n_layers:int=1):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.nb_iter = nb_iter
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.n_layers = n_layers

########################################################################

class GRU(torch.nn.Module):
    """
    model using recurrent neural net architecture
    """
    
    def __init__(self, parameters:GRUParameters):
        super(GRU, self).__init__()
        self.embedding_dim = parameters.embedding_dim
        self.nb_iter = parameters.nb_iter
        self.output_dim = parameters.output_dim
        self.hidden_dim = parameters.hidden_dim
        self.n_layers = parameters.n_layers
        self.drop_prob = parameters.drop_prob
        self.emb = torch.nn.Embedding(parameters.output_dim, parameters.embedding_dim)
        self.gru = torch.nn.GRU(parameters.embedding_dim, parameters.hidden_dim, parameters.n_layers, batch_first=True, dropout=parameters.drop_prob if parameters.n_layers > 1 else 0)
        self.do = torch.nn.Dropout(p=parameters.drop_prob)
        self.fc = torch.nn.Linear(parameters.hidden_dim, parameters.output_dim)
        self.relu = torch.nn.ReLU()
        
###################
        
    def forward(self, x):
        out, h = self.gru(self.emb(x))
        out = self.fc(self.relu(self.do(out)))
        return out, h
    
    def predict(self, word, letter_dict_path):
        letter_dict = json.load(open(letter_dict_path,'r',encoding='utf-8'))
        letters_w = torch.tensor([letter_dict[i] for i in word if i in self.dict])
        out, h = self.forward(letters_w)
        
        
    def train(self, LX, LY):
        nb_iter = self.nb_iter
        optim = torch.optim.Adam(self.parameters())
        lossfunction = torch.nn.CrossEntropyLoss()

        for iteration in range(nb_iter):
            total_loss = 0
            print("Iteration: {}".format(iteration + 1))

            for (x, y) in zip(LX, LY):
                if x.size() == y.size():
                    optim.zero_grad()
                    yprime, h = self.forward(x)
                    loss = lossfunction(yprime, y)
                    total_loss += loss.item()
                    loss.backward()
                    optim.step()

            # Print training loss
            print("Training Loss: {:.4f}".format(total_loss / len(LX)))

        print("Training completed.")

########################################################################



