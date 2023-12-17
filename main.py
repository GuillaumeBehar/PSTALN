from io import open
from conllu import parse
from torch.utils.data import DataLoader
from dataloader import Textdata
from GRU import *

# Chemin vers votre fichier de données et votre fichier de dictionnaire
train_file_path = "UD_French-Sequoia/fr_gsd-ud-train.conllu"
letter_dict_path = "letter_dict_fr.json"

# Création de votre Dataset
dataset = Textdata(train_file_path, letter_dict_path)

print("")
print("Taille du dataset d'entrainement :", len(dataset.data))

# Création du DataLoader
batch_size = 2 # Définissez la taille du batch selon votre besoin
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

vocabsize = len(dataset.dict)
parameters = GRUParameters(letter_dict_path, vocabsize, n_layers=2, is_bidirectional=True)

try :
    model = torch.load("100emb_64hidden_2layer_bidirectional.pt")
except :
    model = GRU(parameters)

print("Nombre de paramètres du model:", model.parameters_number())

#model.train_loop(dataset, 30)
    
