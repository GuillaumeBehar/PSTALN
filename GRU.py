
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

########################################################################

if torch.cuda.is_available():
    device = torch.device("cuda")
else :    
    device = torch.device("cpu")

########################################################################

class GRUParameters():
    """
    Parameters of the GRUNet
    """
    def __init__(self, output_dim:int, hidden_dim:int=64, embedding_dim:int=50, drop_prob:float=0.2, n_layers:int=1):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
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
        pad = letter_dict["#"]
        letters_w = torch.tensor([letter_dict[i] for i in word if i in letter_dict] + [pad] * 6)
        out, h = self.forward(letters_w)
        out = torch.argmax(out, dim=1)
        letter = letter_dict.keys()
        predicted_word = ""
        for index in out :
            predicted_word += letter[int(index)]
        return predicted_word
    
    def validate(self, dataloader):

        self.eval()
        lossfunction = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient computation during validation
            for i, batch in enumerate(dataloader):
                for example in batch:
                    x = example[0, :]
                    y = example[1, :]
                    x = x.to(device)
                    y = y.to(device)
                if x.size() == y.size():
                    yprime, _ = self.forward(x)
                    loss = lossfunction(yprime, y)
                    total_loss += loss.item()
                predicted_labels = torch.argmax(yprime, dim=1)
                correct_predictions += torch.sum(predicted_labels == y).item()
                total_samples += y.size(0)
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    
    def train_one_epoch(self, dataloader):

        self.train(True)
        optim = torch.optim.Adam(self.parameters())
        lossfunction = torch.nn.CrossEntropyLoss()

        for i, batch in enumerate(tqdm(dataloader, desc="Training")):
            for example in batch:
                x = example[0,:]
                y = example[1,:]
                x = x.to(device)
                y = y.to(device)
                if x.size() == y.size():
                    optim.zero_grad()
                    yprime, h = self.forward(x)
                    loss = lossfunction(yprime, y)
                    loss.backward()
                    optim.step()
            

    def train_loop(self, dataset, nb_epoch):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  # Move the model to the GPU if available

        for epoch_number in range(nb_epoch):

            train_subset, val_subset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(1))

            train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=1)
            val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=1)
            
            print('EPOCH {}:'.format(epoch_number + 1))
            self.train_one_epoch(train_loader)

            self.validate(val_loader)

        pass
########################################################################



