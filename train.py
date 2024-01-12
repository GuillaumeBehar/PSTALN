import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from dataProcessing import PosData, oov_proportion, oov_proportion_from_dict
from conllu import parse
from tqdm import tqdm
from UDTagSet import UDTagSet
from models import GRUNet, BiGRUNet, BiLSTMPOSTagger, CNN_GRU


def train(training_data: Dataset,
          valid_data,
          model: nn.Module,
          batch_size: int,
          num_epoch: int,
          learning_rate: float,
          save_name: str
          ) -> None:
    '''
    A compléter au niveau de la validation notamment et séparer form,,label dans l'entraînement.
    '''
    # Create training data
    data = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    validation_perf = []
    eye = torch.eye(19)
    for epoch in range(num_epoch):
        total_loss = 0.0
        num_batches = len(data)
        with tqdm(total=num_batches) as pbar:
            model.train()
            for batch_idx, batch in enumerate(data):
                #print(f"dimension du batch: {batch.shape}")
                # print(f"dimension d'un batch:{batch.shape}")
                x = batch[:, 0, :]
                #print(f"dimension de l'input:{x.shape}")
                y = batch[:, 1]
                y = torch.stack(
                    [torch.stack([eye[pos_idx.item()] for pos_idx in sentence]) for sentence in batch[:, 1, :]])
                #print(f"dimension de l'output:{y.shape}")
                optimizer.zero_grad()
                y_predicted = model.forward(x)
                #print(f"dimension de la sortie:{y_predicted.shape}")
                loss = criterion(input=y_predicted,target=y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
        epoch_loss = total_loss / num_batches
        valid = test(valid_data, model,training_data.get_dict())
        validation_perf.append(valid)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}, valid: {valid}')

    plt.plot(range(1, num_epoch + 1), validation_perf, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Over Epochs')
    plt.show()

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)


def test(test_file, model_path, dictionary):
    sentences = parse(open(test_file, 'r', encoding="utf-8").read())
    dicoUpos = UDTagSet()
    n = 0
    wrong = 0
    for sentence in sentences:
        word_list = [token['form'] for token in sentence]
        # print(word_list)
        y = torch.tensor([dicoUpos[token['upos']] for token in sentence])
        y_predicted = predict_pos(word_list, model_path, dictionary)
        # print(prediction_to_pos(y_predicted))
        wrong += np.count_nonzero(y - y_predicted)
        n += len(sentence)
    return (n - wrong) / n * 100


def predict_pos(sentence, model, dictionary):
    model.eval()
    tokens = [dictionary[word] if word in dictionary else dictionary['<UNK>'] for word in sentence]
    with torch.no_grad():
        input_tensor = torch.LongTensor(tokens)
        output = model(input_tensor)
        y = F.softmax(output, dim=1)
        return torch.argmax(y, dim=1)


def prediction_to_pos(prediction):
    dicoUpos = UDTagSet()
    res = []
    for idx in prediction:
        res.append(dicoUpos.code2tag(idx))
    return res

def train_CNN_GRU(training_data: Dataset,
          valid_data,
          model: nn.Module,
          batch_size: int,
          num_epoch: int,
          learning_rate: float,
          save_name: str
          ) -> None:
    # Create training data
    data = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    validation_perf = []
    eye = torch.eye(19)
    for epoch in range(num_epoch):
        total_loss = 0.0
        num_batches = len(data)
        with tqdm(total=num_batches) as pbar:
            model.train()
            for batch_idx, batch in enumerate(data):
                print(f"dimension du batch: {batch.shape}")
                print(f"dimension d'un batch:{batch.shape}")
                x = batch[:, 0, :]
                print(f"dimension de l'input:{x.shape}")
                y = batch[:, 1]
                y = torch.stack(
                    [torch.stack([eye[pos_idx.item()] for pos_idx in sentence]) for sentence in batch[:, 1, :]])
                print(f"dimension de l'output:{y.shape}")
                optimizer.zero_grad()
                y_predicted = model.forward(x)
                print(f"dimension de la sortie:{y_predicted.shape}")
                loss = criterion(input=y_predicted,target=y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
        epoch_loss = total_loss / num_batches
        valid = test(valid_data, model,training_data.get_dict())
        validation_perf.append(valid)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}, valid: {valid}')

    plt.plot(range(1, num_epoch + 1), validation_perf, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Over Epochs')
    plt.show()

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    training_file = 'files/sequoia-train5.conllu'
    validation_file = 'UD_French/fr_sequoia-ud-dev.conllu'
    test_file = 'UD_French/fr_sequoia-ud-test.conllu'
    train_data = PosData(training_file)
    dico = train_data.get_dict()
    print(f"taille du vocabulaire:{len(dico)}, volume des données:{len(train_data)}")
    model = BiLSTMPOSTagger(vocab_size=len(dico)+1,
                            embedding_dim= 150,
                            hidden_dim=50,
                            output_dim=19,
                            pad_idx=len(dico),
                            dropout=0.1)
    train(train_data,validation_file, model, batch_size=128, num_epoch=40, learning_rate=0.001, save_name='posTagging7.pth')
    
    #model.load_state_dict(torch.load('saved_models/posTagging7.pth'))
    #perf = test(test_file, model, dico)
    #print(f'proportion de oov : {oov_proportion_from_dict(test_file, dico)}')
    #print(f'performance du modèle 1 : {perf}')