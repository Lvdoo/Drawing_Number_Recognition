# FICHE DE RÉVISION — MLP & CNN (Deep Learning) --> -->

---

## PIPELINE GLOBAL (COMMUN À TOUS LES MODÈLES)

---

1. Dataset
   → Données d’entrée (images, labels)
   → Exemple : MNIST (images de chiffres + labels 0–9)

2. Preprocessing
   → Transformer les données pour le modèle
   → ToTensor : image → tensor
   → Normalize : stabilise l’apprentissage

3. DataLoader
   → Crée des minibatchs
   → Permet un entraînement plus rapide et stable

4. Modèle
   → Définit l’architecture du réseau (MLP ou CNN)

5. Forward pass
   → Le modèle fait une prédiction à partir des données

6. Loss function
   → Calcule l’erreur entre prédiction et label
   → Exemple : CrossEntropyLoss
   → Objectif : mesurer “à quel point le modèle se trompe”

7. Backpropagation
   → Calcule les gradients (comment corriger l’erreur)
   → Permet de savoir comment modifier les poids

8. Optimizer
   → Met à jour les poids du modèle
   → Exemple : Adam
   → Formule : nouveaux poids = anciens poids - learning_rate × gradient

9. Training loop
   → Répéter :

   * prédiction
   * calcul de la loss
   * correction des poids

10. Evaluation
    → Tester le modèle sur des données non vues
    → Calcul de l’accuracy

---

## MLP (MULTILAYER PERCEPTRON)

---

Structure :

Input → Flatten → Linear → ReLU → Linear → Output

Étapes :

1. Flatten
   → Transforme image 2D en vecteur
   → Exemple : 28×28 → 784

2. Linear layer
   → Applique une transformation linéaire
   → y = Wx + b

3. ReLU
   → Activation non linéaire
   → Permet d’apprendre des relations complexes
   → ReLU(x) = max(0, x)

4. Output layer
   → Donne un score pour chaque classe

5. Softmax (implicite avec CrossEntropy)
   → Transforme les scores en probabilités

Points clés :

* Simple mais perd la structure de l’image
* Moins performant pour la vision
* Bon pour comprendre les bases

---

## CNN (CONVOLUTIONAL NEURAL NETWORK)

---

Structure :

Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Linear → Output

Étapes :

1. Convolution (Conv2d)
   → Applique des filtres sur l’image
   → Détecte des motifs (bords, formes)

2. Feature maps
   → Résultat des convolutions
   → Représentation des motifs détectés

3. ReLU
   → Introduit la non-linéarité

4. Pooling (MaxPool)
   → Réduit la taille de l’image
   → Garde l’information importante
   → Rend le modèle plus robuste

5. Stacking (plusieurs conv)
   → Permet d’apprendre des motifs de plus en plus complexes

6. Flatten
   → Transforme les features en vecteur

7. Linear layers
   → Prend la décision finale (classification)

Points clés :

* Garde la structure spatiale
* Meilleur pour les images
* Apprend automatiquement les features

---

## DIFFÉRENCE MLP vs CNN

---

MLP :

* Aplati l’image
* Perd les relations spatiales
* Moins performant en vision

CNN :

* Garde la structure 2D
* Apprend des motifs visuels
* Beaucoup plus performant

---

## CONCEPTS IMPORTANTS À MAÎTRISER

---

Loss function
→ Mesure l’erreur du modèle

Gradient
→ Direction pour corriger les poids

Optimizer
→ Applique la correction

Epoch
→ 1 passage complet sur le dataset

Batch
→ Sous-ensemble de données

Overfitting
→ Modèle trop adapté aux données d’entraînement

Generalization
→ Capacité à bien prédire sur de nouvelles données

---

## CHECKLIST POUR CONSTRUIRE UN MODÈLE

---

1. Charger les données
2. Prétraiter les données
3. Créer le modèle
4. Définir la loss
5. Définir l’optimizer
6. Entraîner (training loop)
7. Évaluer
8. Analyser les erreurs
9. Améliorer