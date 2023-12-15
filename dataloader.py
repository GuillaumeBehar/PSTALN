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
        for sentence in self.sentences:
            for token in sentence:
                word = token['form']
                lemma = token['lemma']
                letters_w = [self.dict[i] for i in word if i in self.dict]
                letters_l = [self.dict[i] for i in lemma if i in self.dict]
                lw = len(letters_w)
                ll = len(letters_l)
                if ll<lw:
                    letters_l += [82] * 6
                    letters_w += [82] * (ll + 6 - lw)
                else:
                    letters_w += [82] * 6
                    letters_l += [82] * (lw + 6 - ll)
                examples.append((letters_w,letters_l))
        return examples

    def __getitem__(self, idx):
        return self.data[idx]
