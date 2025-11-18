# PUSL-3123 AI and ML  
*Designed and optimized a Feedforward Multi-Layer Perceptron model with Principal Component Analysis to enhance acceleration-based user authentication using smartwatch data.*

---

## 🎯 Project Overview  
This project implements an authentication approach using smartwatch accelerometer data. We apply data pre-processing, feature reduction (via PCA), and a feedforward Multi-Layer Perceptron (MLP) model to authenticate users based on motion/acceleration patterns.

Key goals:  
- Collect and process acceleration data from smartwatch for different users.  
- Apply signal processing and feature engineering to create meaningful input features.  
- Use Principal Component Analysis (PCA) to reduce dimensionality, remove noise, and improve model performance.  
- Train a feedforward MLP model for user authentication (i.e., classifying whether the data belongs to a specific user).  
- Evaluate performance in terms of accuracy, false accept/reject rates, and computational efficiency.

---

## 📂 Folder Structure  
- `Mathlab files/` — MATLAB code and datasets used for this project.  
  - Data acquisition scripts  
  - Pre-processing routines  
  - PCA implementation and transformation  
  - MLP model definition, training, testing, and evaluation scripts  
- `README.md` — Project description and usage instructions.

---

## 🔧 Getting Started  
### Prerequisites  
- MATLAB (compatible version) 
- Basic knowledge of machine learning (feedforward MLP) and PCA

### How to run  
1. Open the MATLAB script folder.  
2. Load or set the dataset: ensure the acceleration data files are properly referenced.  
3. Run the pre-processing script to clean and normalize data.  
4. Execute the PCA script to reduce feature dimensionality.  
5. Run the MLP training script. Adjust hyperparameters as needed (number of hidden layers/neurons, activation functions, learning rate, epochs, etc.).  
6. After training, run evaluation scripts to measure authentication performance (accuracy, confusion matrix, false accept rate, false reject rate).  
7. Visualise results: feature distributions, PCA variance explained, training/validation loss, ROC curves.

---

## 📊 Key Methods & Techniques  
- **Data Pre-processing**: acceleration data cleaning, normalization, segmentation into windows, feature extraction (e.g., statistical moments, frequency domain features).  
- **Principal Component Analysis (PCA)**: reduces input feature space, retains key variance components, removes redundancy.  
- **Feedforward Multi-Layer Perceptron (MLP)**: standard neural network architecture used here for classification (user vs non-user).  
- **Evaluation Metrics**: accuracy, confusion matrix, FAR (False Accept Rate), FRR (False Reject Rate), ROC/AUC.

---

## ✅ Results & Findings  
- The PCA approach significantly reduced feature dimensionality while maintaining/boosting model performance.  
- Model training time and inference latency were acceptable for a wearable-based authentication scenario.


---

## 📌 Why This Matters  
Wearable devices such as smartwatches offer continuous and unobtrusive user authentication by analysing motion/acceleration signals. By combining PCA (for efficient dimension reduction) with an MLP (for classification), this project demonstrates a viable method for authentication that is less intrusive than traditional methods (passwords, fingerprints) and more adaptable to wearable usage contexts.

---

## 🛠 Customisation & Extensions  
Some ideas for future work or modifications:  
- Expand dataset to more users and longer time spans for better generalisation.  
- Experiment with other dimensionality reduction techniques (e.g., t-SNE, UMAP) or autoencoders.  
- Try alternative network architectures (e.g., CNNs, RNNs) for time-series modelling.  
- Deploy model on a smartwatch or mobile device to test real-time authentication feasibility.  
- Incorporate additional sensors (gyroscope, heart rate) to enhance authentication.

---

## 📧 Contact & Acknowledgements  
Developed as part of the module **PUSL‑3123 AI and ML**.  

---

