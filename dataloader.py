import torch
from torch.utils.data import Dataset
from conllu import parse
import json

class Textdata(Dataset):
    def __init__(self, train_file_path,dict_path):
        self.sentences = parse(open(train_file_path,'r', encoding="utf-8").read())
        self.dict = json.load(open(dict_path,'r',encoding='utf-8'))
        self.data = self.get_listofwords()

    def __len__(self):
        return len(self.sentences)

    def get_listofwords(self):
        examples = [[],[]]
        c=0
        pad = self.dict["#"]
        for sentence in self.sentences:
            for token in sentence:
                word = token['form']
                lemma = token['lemma']
                letters_w = [self.dict[i] for i in word if i in self.dict]
                letters_l = [self.dict[i] for i in lemma if i in self.dict]
                lw = len(letters_w)
                ll = len(letters_l)
                if ll<lw:
                    letters_l += [pad] * 6
                    letters_w += [pad] * (ll + 6 - lw)
                else:
                    letters_w += [pad] * 6
                    letters_l += [pad] * (lw + 6 - ll)
                if len(letters_l) == len(letters_w):
                    examples[0].append(torch.tensor(letters_w))
                    examples[1].append(torch.tensor(letters_l))
                else:
                    print(f'mot ignoré:{word, lemma}')
                    c += 1
        print(f'nombre de mots ignorés: {c}')

        examples[0] = torch.nn.utils.rnn.pad_sequence(examples[0], batch_first=True, padding_value=0)
        examples[1] = torch.nn.utils.rnn.pad_sequence(examples[1], batch_first=True, padding_value=0)

        return [torch.stack((examples[0][k],examples[1][k])) for k in range(len(examples[0]))]

    def __getitem__(self, idx):
        return self.data[idx]
