import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transforme les images en tensors et les normalise
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])

# Importe le dataset en un ensemble de train et test
train_dataset = torchvision.datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transform, download = True)

# Créer des minibatchs
train_load = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_load = DataLoader(dataset = test_dataset, batch_size = 64)

# CNN
class CNN(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(1600,256)
        self.linear2 = nn.Linear(256,10)
    def forward(self,x) :
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = CNN()

# Cross-Entropy loss function (softmax est déjà intégré)
loss_function = nn.CrossEntropyLoss()

# Mets à jour les poids 
optimizer = optim.Adam(params = model.parameters(), lr = 0.001)


# Training
num_epoch = 5 

for epoch in range(num_epoch) : 
    # Lance le model en mode train
    model.train()
    running_loss = 0.0

    for images,labels in train_load : 
        # Reset des gradients
        optimizer.zero_grad()

        # Predictions
        outputs = model(images)

        # loss calcul
        batch_loss = loss_function(outputs,labels)

        # back prop(calcul des gradients de la loss) : permet de savoir dans quel direction modifer les poids
        batch_loss.backward()

        # Mis à jour de poids
        optimizer.step()

        running_loss += batch_loss.item()

    avg_loss = running_loss/len(train_load)
    print(f"Epoch [{epoch+1}/{num_epoch}] - Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "cnn_model.pth")

# Testing

model.eval()
count_answers = 0
total = 0

# Retire les gradients
with torch.no_grad() : 
    for images, labels in test_load :

         # Forward propagation = prédiction
        outputs = model(images)

        # Obtenir le score le plus élevé
        prediction = torch.max(outputs,1)[1]

        # Comparer aux labels et compte le nombre de bonne réponses
        count_answers += (prediction == labels).sum().item()

         # Nombre d'images par minibatch
        total += labels.size(0) #.size(0) permet de récup la première donnée de l'output qui est un tensor qui indique la taille de labels

    accuracy = count_answers/total

print(f"Accuracy : {accuracy*100:.2f}%")
