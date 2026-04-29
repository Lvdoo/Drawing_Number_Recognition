# ✍️ Number Recognition After Drawing

A simple and interactive Python application that recognizes handwritten digits drawn by the user in real time.

The project combines a Convolutional Neural Network (CNN) trained on the MNIST dataset with a Tkinter graphical interface that allows users to draw digits and instantly get predictions.

---

## 🚀 Features

- Draw a digit (0–9) directly on a canvas  
- Automatic image preprocessing  
- Digit prediction using a trained CNN model  
- Simple and intuitive UI  
- Clear canvas functionality  
- Includes both CNN and MLP implementations  

---

## 🧠 Technologies Used

- Python  
- PyTorch  
- Torchvision  
- Tkinter  
- Pillow (PIL)  
- MNIST Dataset  

---

## 📁 Project Structure

```bash
Number_Recognition_After_Drawing/

├── main.py                    # Entry point of the application  
├── interface.py               # Tkinter interface for drawing  
├── model.py                   # CNN architecture + model loading  
├── preprocessing.py           # Image preprocessing logic  
├── number_recognition_CNN.py  # CNN training script  
├── number_recognition_MLP.py  # MLP training script  
├── cnn_model.pth              # Trained model weights  
├── requirements.txt           # Dependencies  
├── .gitignore  
└── README.md  
```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Lvdoo/Number_Recognition_After_Drawing.git
cd Number_Recognition_After_Drawing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it
Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```
---

## ▶️ How to Run

```bash
python main.py
```

1. Draw a number on the canvas
2. Click PREDICTION
3. The model outputs the predicted digit

---

## 🔍 How It Works

### Step-by-step pipeline:
1. User draws a digit on the Tkinter canvas
2. The drawing is captured as an image
3. Preprocessing is applied:
    - Convert to grayscale
    - Detect drawing area (bounding box)
    - Add margins
    - Resize to 28×28 pixels
    - Normalize (same as MNIST dataset)
4. The processed image is passed into the CNN
5. The model predicts the digit (0–9)

### 🧠 Model Architecture (CNN)

The Convolutional Neural Network includes:

- 2 Convolutional layers
- ReLU activation
- MaxPooling layers
- Flatten layer
- 2 Fully Connected layers
- Output layer (10 classes)

Training setup:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Dataset: MNIST

The trained model is saved as:

```bash
cnn_model.pth
```

### 🏋️‍♂️ Training the Model
To retrain the CNN:

```bash
python number_recognition_CNN.py
```

To test a simpler model:

```bash
python number_recognition_MLP.py
```
---

## ⚠️ Limitations
- The model is trained on MNIST → it expects clean, centered digits
- Performance drops if:
    - Drawing is too small or off-center
    - Stroke is too thin
    - Style differs from MNIST

### Common confusions:
4 ↔ 9
3 ↔ 8
1 ↔ 7

---

## 📈 Possible Improvements

- Real-time prediction while drawing
- Display prediction confidence (softmax probabilities)
- Improve preprocessing (centering, thickness normalization)
- Add custom dataset (your own handwriting)
- Improve UI/UX design
- Avoid saving temporary images (optimize pipeline)
- Add demo GIF or screenshots
- Better separation between training and inference code

---

## 🧑‍💻 Author

Project created by Lvdoo

---

## ⭐ Notes

This project is a strong introduction to:

- Deep Learning (CNNs)
- Computer Vision basics
- End-to-end ML pipeline (training → inference → UI)

Perfect as a first real AI project before moving to more advanced topics like:

- Real-time detection
- Object tracking
- Gesture recognition