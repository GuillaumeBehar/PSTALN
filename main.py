from io import open
from conllu import parse
from torch.utils.data import DataLoader
from dataloader import Textdata
from GRU import *

# Chemin vers votre fichier de données et votre fichier de dictionnaire
train_file_path = "UD_French-Sequoia/fr_sequoia-ud-train.conllu"
letter_dict_path = "letter_dict_fr.json"

# Création de votre Dataset
dataset = Textdata(train_file_path, letter_dict_path)

print("")
print("Taille du dataset d'entrainement :", len(dataset))

# Création du DataLoader
batch_size = 1 # Définissez la taille du batch selon votre besoin
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("")
# Parcours des données chargées par le DataLoader pour vérifier
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1} :")
    print(batch)  # Affichez ici votre batch pour vérifier son contenu
    if i == 4:  # Arrêtez l'exécution après quelques batches pour éviter trop d'impression
        break
print("")

##################################################

vocabsize = len(dataset.dict)+1
parameters = GRUParameters(vocabsize)
model = GRU(parameters)
model.train_loop(dataset, 1)