from io import open
from conllu import parse_incr, parse
from dataloader import Textdata
from GRU import *
import torch

data_file = open("UD_French-Sequoia/fr_sequoia-ud-test.conllu", "r", encoding="utf-8").read()
print(type(data_file))

#for tokenlist in parse_incr(data_file):
#print(tokenlist)

sentences = parse(data_file)
print(sentences[1])

#exemples = []
#for sentence in sentences:
#    for token in sentence:
#        word = token['form']
#        lemma = token['lemma']
#        exemples.append((word, lemma))

from torch.utils.data import DataLoader

# Chemin vers votre fichier de données et votre fichier de dictionnaire
train_file_path = "UD_French-Sequoia/fr_sequoia-ud-test.conllu"
dict_path = "letter_dict_fr.json"

# Création de votre Dataset
dataset = Textdata(train_file_path, dict_path)

print(len(dataset))
# Création du DataLoader
batch_size = 1 # Définissez la taille du batch selon votre besoin
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Parcours des données chargées par le DataLoader pour vérifier
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1} :")
    print(batch)  # Affichez ici votre batch pour vérifier son contenu
    if i == 4:  # Arrêtez l'exécution après quelques batches pour éviter trop d'impression
        break

##################################################

vocabsize = len(dataset.dict)+1
print(vocabsize)
parameters = GRUParameters(vocabsize)
model = GRU(parameters)
model.to(device)

LX = [torch.tensor(a[0]) for a in dataset[:2000]]
LY = [torch.tensor(a[1]) for a in dataset[:2000]]
model.train(LX,LY)
out, h = model.forward(LX[0])
print(out.size())