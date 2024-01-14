import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from dataProcessing import PosData, oov_proportion, oov_proportion_from_dict, PosDataforCNN
import json
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
          save_name: str,
          character_dict
          ) -> None:
    '''
    À compléter au niveau de la validation notamment et séparer formes et labels dans l'entraînement.
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
                #print(batch)
                forms = batch[0].squeeze(0)
                targets = batch[1].squeeze(0)
                #print(f"dimension de forms:{forms.shape}")
                #print(f"dimension de targets: {targets.shape}")
                x = forms
                y = torch.stack(
                    [torch.stack([eye[pos_idx.item()] for pos_idx in targets])])
                y = y.squeeze(0)
                #print(f"dimension du target:{y.shape}")
                optimizer.zero_grad()
                y_predicted, _ = model.forward(x)
                #print(f"dimension de la sortie:{y_predicted.shape}")
                loss = criterion(y_predicted,y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
        epoch_loss = total_loss / num_batches
        valid_acc, valid_f1 = test_CNN_GRU(valid_data, model,training_data.get_dict(),character_dict)
        validation_perf.append(valid_acc)
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}')
        print(f'validation accuracy: {valid_acc}, f1 score: {valid_f1}\n')

    plt.plot(range(1, num_epoch + 1), validation_perf, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy Over Epochs')
    plt.show()

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)

def test_CNN_GRU(test_file, model_path, dictionary, char_dictionary):
    sentences = parse(open(test_file, 'r', encoding="utf-8").read())
    dico_upos = UDTagSet()
    y_true = []
    y_pred = []
    for sentence in sentences:
        word_list = [token['form'] for token in sentence]
        y_true.extend([dico_upos[token['upos']] for token in sentence])
        y_predicted = predict_pos_from_char(word_list, model_path, dictionary, char_dictionary)
        y_pred.extend(y_predicted)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1

def predict_pos_from_char(sentence, model, word_dict, char_dict):
    model.eval()
    tokens = []
    for word in sentence:
        characters = [char_dict[i] if i in char_dict else char_dict["U+FFD"] for i in word]
        if  word in word_dict:
            word_idx = word_dict[word]
        else:
            word_idx = word_dict['<UNK>']
        tokens.append(torch.tensor([word_idx]+characters))
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    with torch.no_grad():
        input_tensor = torch.LongTensor(padded_tokens)
        output, _ = model(input_tensor)
        y = F.softmax(output, dim=1)
        return torch.argmax(y, dim=1)

if __name__ == '__main__':
    training_file = 'files/sequoia-train10.conllu'
    validation_file = 'UD_French/fr_sequoia-ud-dev.conllu'
    test_file = 'UD_French/fr_sequoia-ud-test.conllu'
    with open('files/letter_dict_fr.json', 'r', encoding='utf-8') as f:
        char_dict = json.load(f)
    train_data = PosDataforCNN(test_file,char_dict) #a modif
    dico = train_data.get_dict()
    with open('chemin_fichier_json', "w") as fichier:
        json.dump(dico, fichier)
    print(f"taille du vocabulaire:{len(dico)}, volume des données:{len(train_data)}")
    model = CNN_GRU(char_vocab_size=len(char_dict)+1,
                    char_embedding_dim=50,
                    n_filters=30,
                    vocab_size=len(dico) + 1,
                    word_embedding_dim=70,
                    hidden_dim=50,
                    output_dim=19,
                    n_layers=1,
                    drop_out=0.2,
                    char_drop_out=0.1,
                    padding=2)

    train_CNN_GRU(train_data,
                  validation_file,
                  model,
                  batch_size=1,
                  num_epoch=30,
                  learning_rate=0.001,
                  save_name='posTagging8.pth',
                  character_dict=char_dict)
    
    #model.load_state_dict(torch.load('saved_models/posTagging7.pth'))
    #perf = test(test_file, model, dico)
    #print(f'proportion de oov : {oov_proportion_from_dict(test_file, dico)}')
    #print(f'performance du modèle 1 : {perf}')