import torch
from torch.utils.data import Dataset, DataLoader
from conllu import parse, parse_incr, TokenList
from torch.nn.utils.rnn import pad_sequence
import json
import collections
from UDTagSet import UDTagSet

class LemmaData(Dataset):
    def __init__(self,
                 train_file_path,
                 dict_path):
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

    examples = pad_sequence(forms, batch_first=True, padding_value=pad)
    labels = pad_sequence(lemmas, batch_first=True, padding_value=pad)

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
    word_pad = len(dicoVocab)
    pos_pad = dicoUpos["PAD"]
    forms =[]
    labels =[]
    for sentence in sentences:
        words = []
        pos = []
        c = 0
        for token in sentence :
            words.append(dicoVocab[token['form']])
            pos.append(dicoUpos[token['upos']])
            c += 1
            if c == 10:
                forms.append(torch.tensor(words))
                labels.append(torch.tensor(pos))
                words = []
                pos = []
                c = 0
        if c > 2:
            words.extend([word_pad]*(10-c))
            pos.extend([pos_pad]*(10-c))
            forms.append(torch.tensor(words))
            labels.append(torch.tensor(pos))
    data = [torch.stack((forms[k], labels[k])) for k in range(len(forms))]

    return data, dicoVocab

class PosDataforCNN(Dataset):
    def __init__(self, train_file_path, char_dict):
        self.sentences = parse(open(train_file_path, 'r', encoding="utf-8").read())
        self.data, self.word_dict = get_pos_char_List(self.sentences, char_dict, UDTagSet())

    def __len__(self):
        formseq,labelseq = self.data
        return len(formseq)

    def __getitem__(self, idx):
        formseq, labelseq = self.data
        return (formseq[idx],labelseq[idx])

    def get_dict(self):
        return self.dict


def get_pos_char_List(sentences,char_dict, pos_dict):
    word_dict = collections.defaultdict(lambda: len(word_dict))
    char_pad = '#'
    word_pad = len(word_dict)
    pos_pad = pos_dict["PAD"]
    forms =[]
    labels =[]
    for sentence in sentences:
        words = []
        pos = []
        c = 0
        for token in sentence:
            word = token['form']
            char = [char_dict[i] if i in char_dict else char_dict["U+FFD"] for i in word]
            words.append(torch.tensor([word_dict[word]]+char))
            pos.append(pos_dict[token['upos']])
            c += 1
            if c == 10:
                padded_words = pad_sequence(words, batch_first=True, padding_value=char_dict[char_pad])
                forms.append(padded_words)
                labels.append(torch.tensor(pos))
                words = []
                pos = []
                c = 0
        if c > 2:
            words.extend([torch.tensor([word_pad])] * (10 - c))
            pos.extend([pos_pad] * (10 - c))
            padded_words = pad_sequence(words, batch_first=True, padding_value=char_dict[char_pad])
            forms.append(padded_words)
            labels.append(torch.tensor(pos))

    #data = [torch.stack((forms[k], labels[k])) for k in range(len(forms))]

    return (forms, labels), word_dict


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


class LemmaPosData(Dataset):
    def __init__(self,
                 train_file_path):
            self.sentences = parse(open(train_file_path, 'r', encoding="utf-8").read())
            self.data = get_sentenceList(self.sentences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_sentenceList(sentences):
    forms = []
    lemmas = []
    c = 0
    for sentence in sentences:
        words = []
        lemma = []
        for token in sentence:
            words.append(token['form'])
            lemma.append(token['lemma'])
        forms.append(words)
        lemmas.append(lemma)

    pad = 'PAD'
    examples = torch.nn.utils.rnn.pad_sequence(forms, batch_first=True, padding_value=pad)
    labels = torch.nn.utils.rnn.pad_sequence(lemmas, batch_first=True, padding_value=pad)
    return [torch.stack((examples[k], labels[k])) for k in range(len(examples))]


if __name__== '__main__':
    #file = 'UD_French/fr_sequoia-ud-train.conllu'
    #replaceRareWords(file, 10,'sequoia-train10.conllu')

    train_file_path = "UD_French/fr_sequoia-ud-test.conllu"  # Remplacez par le chemin de votre fichier
    with open('files/letter_dict_fr.json', 'r', encoding='utf-8') as f:
        char_dict = json.load(f)

    dataset = PosDataforCNN(train_file_path, char_dict)
    print(len(dataset))
    print(dataset[5])
    #print(dataset.get_dict())

