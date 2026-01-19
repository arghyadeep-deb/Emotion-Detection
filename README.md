# Emotion Detection  
A Deep Learning–based application that detects human emotions from facial images using Convolutional Neural Networks (CNN).  
The project focuses on image-based emotion classification and demonstrates an end-to-end deep learning workflow, including training, evaluation, and model versioning using Git LFS.
---

## About the Project  
This project implements a facial emotion recognition system using a CNN trained on labeled facial image data.  
It classifies human emotions such as **Happy, Sad, Angry, Fear, Surprise, Neutral**, etc., based on visual facial features.
The trained model is saved in `.keras` format and managed using **Git Large File Storage (Git LFS)** to handle large model artifacts professionally.

---

## Features  
- Facial emotion detection using Deep Learning  
- Convolutional Neural Network (CNN) architecture  
- Image preprocessing and normalization  
- Multi-class emotion classification  
- Model training and evaluation using Jupyter Notebook  
- Saved trained model (`.keras`) for reuse and deployment  
- Git LFS integration for large model file management  
- Clean and modular project structure  
---

## Tech Stack  
- **Language:** Python  
- **Deep Learning:** TensorFlow, Keras    
- **Data Processing:** NumPy  
- **Visualization:** Matplotlib  
- **Version Control:** Git, Git LFS  
---

## Project Structure  

```
Emotion-Detection/
│
├── train/                         # Training dataset (emotion-wise folders)
├── test/                          # Testing dataset
│
├── model.ipynb                    # Model training & evaluation notebook
├── emotion_model_final.keras      # Trained CNN model (tracked via Git LFS)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Ignored files & folders
├── .gitattributes                 # Git LFS configuration
└── README.md
```
---

## How to Run the Project  
### Step 1: Clone the repository
```
git clone https://github.com/arghyadeep-deb/Emotion-Detection.git
cd Emotion-Detection
```
---

### Step 2: Create and activate a virtual environment  
```
python -m venv emotionenv
```
**Windows**
```
emotionenv\Scripts\activate
```
**Mac/Linux**
```
source emotionenv/bin/activate
```
---
### Step 3: Install dependencies  
```
pip install -r requirements.txt
```
---
### Step 4: Run the training notebook  
```
jupyter notebook model.ipynb
```
Follow the notebook cells to:
- Load the dataset  
- Train the CNN  
- Evaluate performance  
- Save the trained model  
---

## Using the Trained Model  
```
from tensorflow.keras.models import load_model
model = load_model("emotion_model_final.keras")
predictions = model.predict(image_array)
```
---

## Dataset  
The dataset consists of labeled facial images categorized by emotion classes.  
It enables the CNN to learn discriminative facial features for accurate emotion classification.
---

## Purpose  
This project is intended for learning and demonstrating:
- Deep Learning with CNNs  
- Facial emotion recognition using images  
- Image preprocessing techniques  
- Model saving and reuse  
- Git LFS usage for large ML artifacts  
- Structuring an internship-ready ML project  
---

## Notes  
- The trained model file exceeds GitHub’s 100 MB limit and is handled using **Git Large File Storage (LFS)**
  ```
  git lfs pull
  ```
- This reflects real-world DL project practices  
