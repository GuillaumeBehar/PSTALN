
import torch
import json
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

########################################################################

class GRUParameters:
    """
    Parameters of the GRU Net.
    """
    # Example usage:
    # gru_params = GRUParameters(output_dim=10, hidden_dim=128, embedding_dim=64, drop_prob=0.3, n_layers=2)
    def __init__(self, letter_dict_path: str, output_dim: int, hidden_dim: int = 64, embedding_dim: int = 50, drop_prob: float = 0.2, n_layers: int = 1):
        """
        Initializes the GRUParameters instance.

        Args:
            letter_dict_path (str): The path to the letter dictionary.
            output_dim (int): The dimension of the output.
            hidden_dim (int, optional): The dimension of the hidden state. Defaults to 64.
            embedding_dim (int, optional): The dimension of the input embeddings. Defaults to 50.
            drop_prob (float, optional): Dropout probability. Defaults to 0.2.
            n_layers (int, optional): Number of GRU layers. Defaults to 1.
        """
        self.letter_dict_path = letter_dict_path
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.n_layers = n_layers



########################################################################

class GRU(torch.nn.Module):
    """
    Model using recurrent neural net architecture.
    """
    # Example usage:
    # gru_params = GRUParameters(output_dim=10, hidden_dim=128, embedding_dim=64, drop_prob=0.3, n_layers=2)
    # gru_model = GRU(parameters=gru_params)
    def __init__(self, parameters: GRUParameters):
        """
        Initializes the GRU instance.

        Args:
            parameters (GRUParameters): Parameters for configuring the GRU model.
        """
        super(GRU, self).__init__()

        # Extracting parameters from the provided object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.letter_dict_path = parameters.letter_dict_path
        self.embedding_dim = parameters.embedding_dim
        self.output_dim = parameters.output_dim
        self.hidden_dim = parameters.hidden_dim
        self.n_layers = parameters.n_layers
        self.drop_prob = parameters.drop_prob

        # Define model components
        self.emb = torch.nn.Embedding(parameters.output_dim, parameters.embedding_dim)
        self.gru = torch.nn.GRU(
            parameters.embedding_dim,
            parameters.hidden_dim,
            parameters.n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=parameters.drop_prob if parameters.n_layers > 1 else 0
        )
        self.do = torch.nn.Dropout(p=parameters.drop_prob)
        self.fc = torch.nn.Linear(parameters.hidden_dim, parameters.output_dim)
        self.relu = torch.nn.ReLU()
        
###################
        
    def parameters_number(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
        
    def forward(self, x):
        """
        Defines the forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Hidden state tensor.
        """
        # Apply embedding layer to input
        embedded = self.emb(x)

        # Apply GRU layer
        out, h = self.gru(embedded)

        # Apply dropout, ReLU activation, and linear layer
        out = self.fc(self.relu(self.do(out)))

        return out, h
    
    def predict(self, word):
        """
        Predicts the output word given an input word using the trained GRU model.

        Args:
            word (str): Input word for prediction.
            letter_dict_path (str): Path to the JSON file containing the letter dictionary.

        Returns:
            str: Predicted word.
        """
        # Load letter dictionary from JSON file
        with open(self.letter_dict_path, 'r', encoding='utf-8') as file:
            letter_dict = json.load(file)

        # Get padding value from letter dictionary
        pad = letter_dict.get("#", 0)

        # Convert input word to tensor using letter dictionary
        letters_w = torch.tensor([letter_dict.get(i, pad) for i in word] + [pad] * 6)
        letters_w = letters_w.to(self.device)

        # Forward pass through the GRU model
        out, h = self.forward(letters_w)

        # Find the index with the highest probability in the output tensor
        out = torch.argmax(out, dim=1)

        # Convert indices back to letters using the letter dictionary
        letter = list(letter_dict.keys())
        predicted_word = "".join([letter[int(index)] for index in out])
        return predicted_word
    
    def validate(self, dataloader):
        """
        Validates the GRU model on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for validation.

        Returns:
            float: Average validation loss.
            float: Validation accuracy.
        """
        # Set the model to evaluation mode
        self.eval()

        # Define the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient computation during validation
            for _, batch in enumerate(dataloader):
                for example in batch:
                    x, y = example[0].to(self.device), example[1].to(self.device)

                    # Ensure input and target sizes match
                    if x.size(0) == y.size(0):
                        # Forward pass
                        y_prime, _ = self.forward(x)

                        # Calculate loss
                        loss = loss_function(y_prime, y)
                        total_loss += loss.item()

                        # Calculate accuracy
                        predicted_labels = torch.argmax(y_prime, dim=1)
                        correct_predictions += torch.sum(predicted_labels == y).item()

                        total_samples += y.size(0)

        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # Print validation results
        print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        return avg_loss, accuracy
    
    def train_one_epoch(self, dataloader):
        """
        Trains the GRU model for one epoch on the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for training.
        """
        # Set the model to training mode
        self.train(True)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters())
        loss_function = torch.nn.CrossEntropyLoss()

        # Iterate over batches in the dataloader
        for i, batch in enumerate(tqdm(dataloader, desc="Training")):
            for example in batch:
                x, y = example[0].to(self.device), example[1].to(self.device)

                # Ensure input and target sizes match
                if x.size(0) == y.size(0):
                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    y_prime, h = self.forward(x)

                    # Calculate loss
                    loss = loss_function(y_prime, y)

                    # Backward pass and optimization step
                    loss.backward()
                    optimizer.step()
            

    def train_loop(self, dataset, nb_epoch):
        """
        Training loop for the GRU model.

        Args:
            dataset (torch.utils.data.Dataset): Dataset for training.
            nb_epoch (int): Number of training epochs.
        """
        # gru_model = GRU(parameters=gru_params)
        # gru_model.train_loop(dataset=your_dataset, nb_epoch=10)
        
        self.to(self.device)  # Move the model to the GPU if available

        data = dataset.data

        for epoch_number in range(nb_epoch):
            # Split the dataset into training and validation subsets
            train_subset, val_subset = torch.utils.data.random_split(data, [int(len(data) * 0.9), len(data) - int(len(data) * 0.9)])

            # Create dataloaders for training and validation
            train_loader = DataLoader(dataset=train_subset, shuffle=True, batch_size=1)
            val_loader = DataLoader(dataset=val_subset, shuffle=False, batch_size=1)

            print(f'EPOCH {epoch_number + 1}:')
            
            # Train for one epoch
            self.train_one_epoch(train_loader)

            # Validate on the validation subset
            self.validate(val_loader)

########################################################################



