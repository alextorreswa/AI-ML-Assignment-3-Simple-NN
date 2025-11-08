# MNIST Digit Classification with Feedforward Neural Network
**Author:** Alex Torres  
**Framework:** TensorFlow / Keras  

---

## Project Overview
This project demonstrates the use of a simple **Feedforward Neural Network (FNN)** for handwritten digit classification using the **MNIST dataset** (28×28 grayscale images of digits 0–9).  
The model was built, trained, and evaluated using **TensorFlow/Keras**.

---

## Model Architecture
| Layer | Type | Neurons | Activation | Description |
|-------|------|----------|-------------|--------------|
| Input | Flatten | 784 (28×28) | — | Converts 2D image into a 1D vector |
| Hidden 1 | Dense | 128 | ReLU | Learns non-linear features |
| Hidden 2 | Dense | 64 | ReLU | Adds depth for better representation |
| Output | Dense | 10 | Softmax | Produces probability distribution over 10 classes |

- **Optimizer:** Adam  
- **Loss Function:** Categorical Cross-Entropy  
- **Metric:** Accuracy  

---

## Data Preparation
- Loaded the **MNIST** dataset from `tf.keras.datasets`.  
- Normalized pixel values to the range **[0, 1]**.  
- Flattened each image into a **784-dimensional** vector.  
- One-hot encoded the output labels (digits 0–9).

---

## Training Configuration
- **Epochs:** 8  
- **Batch size:** 32  
- **Validation split:** 10%  
- **Activation functions:**  
  - **ReLU** for hidden layers → adds non-linearity and avoids vanishing gradients.  
  - **Softmax** for output → converts logits into class probabilities.

---

## Final Model Performance
| Metric | Result |
|--------|---------|
| **Test Accuracy** | **0.9758 (97.6%)** |
| **Test Loss** | 0.0818 |

---

## Example Prediction
For a random test image:  
- **True label:** 2  
- **Predicted label:** 2  
- **Top-3 probabilities:**  
  - Digit 2 → 0.960  
  - Digit 3 → 0.037  
  - Digit 7 → 0.003  

The model confidently recognized the handwritten digit with high probability.
