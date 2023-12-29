import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataProcessing import PosData
from conllu import parse, parse_incr, TokenList
from UDTagSet import UDTagSet
from models import GRUNet

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
        for batch_idx, batch in enumerate(data):
            #print(f"dimension d'un batch:{batch.shape}")
            x = batch[:,0,:]
            #print(f"dimension de l'input:{x.shape}")
            y = torch.stack([torch.stack([eye[pos_idx.item()] for pos_idx in sentence])for sentence in batch[:,1,:]])
            #print(f"dimension de l'output:{y.shape}")

            y_predicted,h = model.forward(x)
            #print(f"dimension de la sortie:{y_predicted.shape}")
            loss = lossFct(y_predicted, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epoch}], batch [{batch_idx + 1}/{num_batches}]')
        epoch_loss = total_loss / num_batches
        print(f'Epoch [{epoch + 1}/{num_epoch}], Avg loss: {epoch_loss}')

    # saving model
    save_path = 'saved_models/' + save_name
    torch.save(model.state_dict(), save_path)

def test(test_file, model_path, dictionary):
    sentences = parse(open(test_file,'r', encoding="utf-8").read())
    n = 0
    wrong = 0
    for sentence in sentences:
        word_list = [token['form'] for token in sentence]
        y = torch.tensor([token['upos'] for token in sentence])
        y_predicted = predict_pos(word_list,model_path,dictionary)
        wrong += np.count_nonzero(y-y_predicted)
        n += len(sentence)
    return wrong/n


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


if __name__=='__main__':
    training_file = 'files/train10.conllu'
    train_data = PosData(training_file)
    dico = dict(train_data.get_dict())
    model = GRUNet(input_dim=100, hidden_dim=50,output_dim=19, vocabSize= len(dico)+1)
    train(train_data, model, batch_size=10, num_epoch=5, learning_rate=0.01, save_name='posTagging1.pth')
    model.load_state_dict(torch.load('saved_models/posTagging1.pth'))
    test_sentence = ['je','veux', 'un', 'gouvernement', 'qui', 'est', 'bien', 'mieux', 'que', 'celui', 'de', 'la', 'France']
    pos_prediction = predict_pos(test_sentence,model,dico)
    print(prediction_to_pos(pos_prediction))
    #train(train_data,model,batch_size=10,num_epoch=3,learning_rate=0.01,save_name='posTagging0.pth')
