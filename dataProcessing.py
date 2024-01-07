import torch
from torch.utils.data import Dataset
from conllu import parse, parse_incr, TokenList
import json
import collections
from UDTagSet import UDTagSet

class LemmaData(Dataset):
    def __init__(self, train_file_path,dict_path):
        self.sentences = parse(open(train_file_path,'r', encoding="utf-8").read())
        self.dict = json.load(open(dict_path,'r',encoding='utf-8'))
        self.data = get_lemmaList(self.sentences, self.dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_lemmaList(sentences, char_dict):
    forms = []
    lemmas = []
    c = 0
    pad = char_dict["#"]
    for sentence in sentences:
        for token in sentence:
            word = token['form']
            lemma = token['lemma']
            letters_w = [char_dict[i] for i in word if i in char_dict]
            letters_l = [char_dict[i] for i in lemma if i in char_dict]
            lw = len(letters_w)
            ll = len(letters_l)
            if ll < lw:
                letters_l += [pad] * 6
                letters_w += [pad] * (ll + 6 - lw)
            else:
                letters_w += [pad] * 6
                letters_l += [pad] * (lw + 6 - ll)
            if len(letters_l) == len(letters_w):
                forms.append(torch.tensor(letters_w))
                lemmas.append(torch.tensor(letters_l))
            else:
                print(f'mot ignoré:{word, lemma}')
                c += 1
    print(f'nombre de mots ignorés: {c}')

    examples = torch.nn.utils.rnn.pad_sequence(forms, batch_first=True, padding_value=pad)
    labels = torch.nn.utils.rnn.pad_sequence(lemmas, batch_first=True, padding_value=pad)

    return [torch.stack((examples[k], labels[k])) for k in range(len(examples))]



class PosData(Dataset):
    def __init__(self, train_file_path):
        self.sentences = parse(open(train_file_path,'r', encoding="utf-8").read())
        self.data,self.dict = get_posList(self.sentences,UDTagSet())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dict(self):
        return self.dict

def get_posList(sentences, dicoUpos) :
    dicoVocab = collections.defaultdict(lambda: len(dicoVocab))
    forms =[]
    parts_os =[]
    c = 0
    for sentence in sentences:
        words = []
        pos = []
        c+=1
        for token in sentence :
            words.append(dicoVocab[token['form']])
            pos.append(dicoUpos[token['upos']])
        if len(words) != len(pos):
            print(f'problème à la phrase {c}')
        forms.append(torch.tensor(words))
        parts_os.append(torch.tensor(pos))

    word_pad = len(dicoVocab)
    pos_pad = dicoUpos["PAD"]
    examples = torch.nn.utils.rnn.pad_sequence(forms, batch_first=True, padding_value=word_pad)
    labels = torch.nn.utils.rnn.pad_sequence(parts_os, batch_first=True, padding_value=pos_pad)
    data = [torch.stack((examples[k], labels[k])) for k in range(len(forms))]

    return data, dicoVocab


def replaceRareWords(conlluFileName, wordThreshold, newfileName):
    wordCounts = {}
    wordsTotal = 0

    # Faire des statistiques sur les caractères et les mots
    with open(conlluFileName, "r", encoding="utf-8") as data_file:
        for sentence in parse_incr(data_file):
            for token in sentence:
                wordsTotal += 1
                form = token['form']
                wordCounts[form] = wordCounts.get(form, 0) + 1

    wordsReplaced = 0
    with open(newfileName, "w", encoding="utf-8") as new_file:
        with open(conlluFileName, "r", encoding="utf-8") as data_file:
            for sentence in parse_incr(data_file):
                for token in sentence:
                    form = token['form']
                    if wordCounts[form] < wordThreshold:
                        token['form'] = '<UNK>'
                        wordsReplaced += 1
                        if token['misc']:
                            token['misc']['OrigForm'] = form
                        else:
                            token['misc'] = {"OrigForm": form}

                new_file.write(TokenList(sentence).serialize() + '\n')
    print(f'nombre de mots remplacés par <UNK>:{wordsReplaced}')

def oov_proportion(file):
    total = 0
    oov = 0
    with open(file, "r", encoding="utf-8") as corpus:
        for sentence in parse_incr(corpus):
            for token in sentence:
                total += 1
                if token['form'] == '<UNK>':
                    oov += 1
    return oov/total * 100

def oov_proportion_from_dict(file,dictionary):
    total = 0
    oov = 0
    with open(file, "r", encoding="utf-8") as corpus:
        for sentence in parse_incr(corpus):
            for token in sentence:
                total += 1
                if not token['form'] in dictionary:
                    oov += 1
    return oov / total * 100

class EmbForPosData(Dataset):
    def __init__(self, train_file_path, falseWord_proportion = int):
        self.sentences = parse(open(train_file_path,'r', encoding="utf-8").read())
        self.data,self.dict = get_PosForEmbList(self.sentences,UDTagSet())
        self.falseWord_prop = falseWord_proportion

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_dict(self):
        return self.dict

def get_PosForEmbList(sentences, dicoUpos) :
    dicoVocab = collections.defaultdict(lambda: len(dicoVocab))
    words =[]
    pos =[]
    c = 0
    for sentence in sentences:
        c+=1
        for token in sentence :
            words.append(dicoVocab[token['form']])
            pos.append(dicoUpos[token['upos']])
        if len(words) != len(pos):
            print(f'problème à la phrase {c}')
    tokens = torch.tensor(words)
    labels = torch.tensor(pos)
    data = [torch.stack((tokens[k], labels[k])) for k in range(len(words))]
    return data, dicoVocab


if __name__== '__main__':
    train_file_path = "UD_French/fr_gsd-ud-train.conllu"
    newfile = 'files/gsd-train10.conllu'
    word_threshold = 20
    #replaceRareWords(train_file_path,wordThreshold=word_threshold,newfileName=newfile)
    print(oov_proportion(newfile))