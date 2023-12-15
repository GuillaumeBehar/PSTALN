import torch
from torch.utils.data import Dataset, DataLoader
from conllu import parse_incr, parse
import json

class Textdata(Dataset):
    def __init__(self, train_file_path,dict_path):
        self.sentences = parse(open(train_file_path,'r', encoding="utf-8").read())
        self.dict = json.load(open(dict_path,'r',encoding='utf-8'))
        self.data = self.get_listofwords()

    def __len__(self):
        return len(self.sentences)

    def get_listofwords(self):
        examples = []
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
                    examples.append(torch.stack((torch.tensor(letters_w), torch.tensor(letters_l))))
                else:
                    print(f'mot ignoré:{word, lemma}')
                    c += 1
        print(f'nombre de mots ignorés: {c}')
        return examples

    def __getitem__(self, idx):
        return self.data[idx]
