# KidneyCT-ImageClassifier
 The deep learning model leverages MobileNetV2 and transfer learning techniques to efficiently classify kidney conditions into four categories: Normal, Cyst, Stone, and Tumor. The workflow includes image preprocessing, model training, evaluation (accuracy, confusion matrix, classification report), and presentation. 
📌 Project Overview
Domain: Medical Imaging

Objective: To classify kidney CT images into four categories for early diagnosis

Model Used: MobileNetV2 with transfer learning

Tech Stack: Python, TensorFlow/Keras, OpenCV, Scikit-learn

Dataset: CT scan images categorized into Normal, Cyst, Stone, Tumor

🧾 Features
MobileNetV2 pretrained on ImageNet for efficient feature extraction

Input images resized to 128x128 for consistency

Training with early stopping and learning rate reduction

Output includes:

Accuracy score

Confusion matrix

Classification report with precision, recall, and F1-score

🗂️ Dataset Preprocessing
Images were labeled manually by folder names.

Each image is resized using OpenCV.

Pixel values are normalized.

Split into training, validation, and testing sets.

python
Copy code
def load_and_resize(image_path, size=(128, 128)):
    img = imread(str(image_path))
    img = cv2.resize(img, size)
    return img
🏗️ Model Architecture
python
Copy code
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128,128,3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)
Optimizer: Adam

Loss: SparseCategoricalCrossentropy

Metrics: Accuracy

Callbacks:

EarlyStopping

ReduceLROnPlateau

📊 Results
High accuracy on training and validation

Confusion Matrix showed effective classification across all four categories

Evaluation Metrics provided confidence in model performance

python
Copy code
print(confusion_matrix(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels))
🌍 Sustainability Development Goals
SDG 3: Good Health and Well-being

SDG 9: Industry, Innovation and Infrastructure

SDG 10: Reduced Inequalities

🔮 Future Work
Real-time web app using Streamlit or Flask

Add explainable AI (Grad-CAM, LIME)

Integrate clinical metadata

Optimize for edge deployment using quantization

🧰 How to Run
bash
Copy code
# Clone the repository
git clone https://github.com/your-username/Kidney-CT-Classification.git
cd Kidney-CT-Classification

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook mobileNetV4_1.ipynb
📦 Requirements
List this in your requirements.txt:

txt
Copy code
tensorflow
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
👨‍💻 Authors
Mohammed Riyaz Rasool – CSE (AI & ML), Vardhaman College of Engineering
Supervisor: Dr. M. A. Jabbar
