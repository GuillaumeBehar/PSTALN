import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataProcessing import PosData, oov_proportion, oov_proportion_from_dict, EmbForPosData
from conllu import parse
from tqdm import tqdm
from UDTagSet import UDTagSet
from models import GRUNet, BiGRUNet, FeedForward, GRU_prtEmb

def train(training_data, model, batch_size = int, num_epoch=int, learning_rate = float, save_name=str):
    # Create training data
    data = DataLoader(training_data, batch_size= batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    lossFct = torch.nn.CrossEntropyLoss()

    # Training
    eye = torch.eye(19)
    for epoch in range(num_epoch):
        total_loss = 0.0
        num_batches = len(data)
        with tqdm(total=num_batches) as pbar:
            for batch_idx, batch in enumerate(data):
                #print(f"dimension d'un batch:{batch.shape}")
                x = batch[:,0,:]
                #print(f"dimension de l'input:{x.shape}")
                y = torch.stack([torch.stack([eye[pos_idx.item()] for pos_idx in sentence])for sentence in batch[:,1,:]])
                #print(f"dimension de l'output:{y.shape}")
                optimizer.zero_grad()
                y_predicted,h = model.forward(x)
                #print(f"dimension de la sortie:{y_predicted.shape}")
                loss = lossFct(y_predicted, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
        epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}')

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)

def test(test_file, model_path, dictionary):
    sentences = parse(open(test_file,'r', encoding="utf-8").read())
    dicoUpos = UDTagSet()
    n = 0
    wrong = 0
    for sentence in sentences:
        word_list = [token['form'] for token in sentence]
        #print(word_list)
        y = torch.tensor([dicoUpos[token['upos']] for token in sentence])
        y_predicted = predict_pos(word_list,model_path,dictionary)
        #print(prediction_to_pos(y_predicted))
        wrong += np.count_nonzero(y-y_predicted)
        n += len(sentence)
    return (n-wrong)/n*100


def predict_pos(sentence, model, dictionary):
    model.eval()
    tokens = [dictionary[word] if word in dictionary else dictionary['<UNK>'] for word in sentence]
    with torch.no_grad():
        input_tensor = torch.LongTensor(tokens)
        output, _ = model(input_tensor)
        y = F.softmax(output,dim=1)
        return torch.argmax(y,dim=1)

def prediction_to_pos(prediction):
    dicoUpos = UDTagSet()
    res = []
    for idx in prediction:
        res.append(dicoUpos.code2tag(idx))
    return res

def train_emb(training_data, model,batch_size = int, num_epoch=int, learning_rate = float, save_name=str):
    # Create training data
    data = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lossFct = torch.nn.CrossEntropyLoss()

    # Training
    eye = torch.eye(19)
    for epoch in range(num_epoch):
        total_loss = 0.0
        num_batches = len(data)
        with tqdm(total=num_batches) as pbar:
            for batch_idx, batch in enumerate(data):
                #print(f"dimension d'un batch:{batch.shape}")
                x = batch[:, 0]
                #print(f"dimension de l'input:{x.shape}")
                y = torch.stack([eye[pos_idx.item()] for pos_idx in batch[:,1]])
                #print(f"dimension de l'output:{y.shape}")
                optimizer.zero_grad()
                y_predicted = model.forward(x)
                #print(f"dimension de la sortie:{y_predicted.shape}")
                loss = lossFct(y_predicted, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}')
        epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}')

    # Saving the embedding matrix as a CSV file
    embedding_matrix = model.embedding.weight.detach().numpy()
    df = pd.DataFrame(embedding_matrix)
    embedding_file = 'saved_models/' + 'embedding50_matrix.csv'
    df.to_csv(embedding_file, index=False)

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)

def test_emb(test_file, model, dictionary):
    model.eval()
    sentences = parse(open(test_file,'r', encoding="utf-8").read())
    dicoUpos = UDTagSet()
    n = 0
    right = 0
    for sentence in sentences:
        for token in sentence:
            word = token['form']
            input = torch.tensor(dictionary[word] if word in dictionary else dictionary['<UNK>'])
            output = model(input)
            y_ = F.softmax(output,dim =0)
            y = dicoUpos[token['upos']]
            if y == torch.argmax(y_):
                right += 1
            n += 1
    return right/n*100

if __name__=='__main__':
    training_file = 'UD_French/gsd-ud-train.conllu'
    test_file = 'UD_French/fr_gsd-ud-test.conllu'
    train_data = PosData(training_file)
    dico = dict(train_data.get_dict())
    print(f"taille du vocabulaire:{len(dico)}, volume des données:{len(train_data)}")
    model = GRUNet(input_dim = 50, hidden_dim=30,output_dim=19,vocabSize=len(dico)+1)
    train(train_data, model, batch_size=20, num_epoch=20, learning_rate=0.01, save_name='posTagging2.pth')
    model.load_state_dict(torch.load('saved_models/posTagging2.pth'))
    test_sentence = ['je','veux', 'un', 'gouvernement', 'qui', 'est', 'bien', 'mieux', 'que', 'celui', 'de', 'la', 'France']
    pos_prediction = predict_pos(test_sentence,model,dico)
    print(prediction_to_pos(pos_prediction))
    perf = test(test_file,model,dico)
    print(f'proportion de oov : {oov_proportion_from_dict(test_file, dico)}')
    print(f'performance du modèle:{perf}')
