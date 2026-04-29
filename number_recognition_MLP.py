import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transforme les images en tensors et normalise les valeurs
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

# Charge le dataset en un ensemble de train et de test
train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transform, download = True)

#Créer des minibatchs
load_train = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
load_test = DataLoader(dataset = test_dataset, batch_size = 64)

## Voir infos basiques
# print(len(train_dataset), len(test_dataset))
# image,label = train_dataset[0]
# print("shape : ", image.shape)
# print("label", label)

# images, labels = next(iter(load_train))

# print("Batch images shape:", images.shape)
# print("Batch labels shape:", labels.shape)


## Vérifie le nombre de labels
# unique_labels = set()

# for _, label in train_dataset:
#     unique_labels.add(label)
#     if len(unique_labels) == 10:
#         break

# print("Labels trouvés :", unique_labels)

# MLP et forward propagation
class MLP(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.model = nn.Sequential(nn.Flatten(), nn.Linear(784,256), nn.ReLU(), nn.Linear(256,10))
    def forward(self,x) : 
        return self.model(x)

model = MLP()

# Cross-Entropy loss function (softmax est déjà intégré)
loss_function = nn.CrossEntropyLoss()

# Mets à jour les poids 
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training boucle
num_epoch = 5

for epoch in range(num_epoch) :
    # lance le model en mode train
    model.train()
    running_loss = 0.0

    for images,labels in load_train :

        # reset les gradients
        optimizer.zero_grad()

        # forward propagation = prédiction
        outputs = model(images)

        # loss calcul
        batch_loss = loss_function(outputs,labels)

        # back prop(calcul des gradients de la loss) : permet de savoir dans quel direction modifer les poids
        batch_loss.backward()

        # met à jour les poids 
        optimizer.step()

        running_loss += batch_loss.item()

    avg_loss = running_loss/len(load_train)
    print(f"Epoch [{epoch+1}/{num_epoch}] - Loss: {avg_loss:.4f}")


#Evaluation du model
model.eval()
count_answers = 0
total = 0

# Coupe les gradients car pas beosin
with torch.no_grad() :
    for images,labels in load_test :

        # Forward propagation = prédiction
        outputs = model(images)

        # Obtenir le score le plus élevé
        prediction = torch.max(outputs,1)[1]

        # Comparer aux labels et compte le nombre de bonne réponses
        count_answers += (prediction == labels).sum().item()

        # Nombre d'images par minibatch
        total += labels.size(0) #.size(0) permet de récup la première donnée de l'output qui est un tensor qui indique la taille de labels

        # Permet d'afficher des images, leur prédiction et leur label
        for i in range(3):
            image = images[i].squeeze() #.squeeze() permet de retirer les dimensions de taille 1, ici image=[1,28,28], on retire le 1 du grayscale

            # dénormalisation pour affichage
            image = image * 0.3081 + 0.1307

            plt.imshow(image, cmap="gray")
            plt.title(f"Vrai label : {labels[i].item()} | Prédiction : {prediction[i].item()}")
            plt.axis("off")
            plt.show()


    # Accuracy 
    accuracy = count_answers / total

print(f"Accuracy : {accuracy*100:.2f}%")

