
# Hopfield Network for MNIST Digit Recognition

## 📌 Overview

This project implements a **Hopfield Neural Network** to store and retrieve prototype patterns of handwritten digits (**0–9**) from the MNIST dataset.
The network demonstrates **associative memory** by reconstructing original digit patterns from noisy inputs, showcasing its **error-correction capabilities**.

---

## 🎯 Purpose

The code:

* Creates **prototype patterns** for each digit (0–9) using **50 MNIST samples per digit**.
* Builds a **Hopfield network weight matrix** to store these patterns.
* Tests the network by adding **20% pixel-flip noise**, then retrieves the closest stored pattern and predicts the digit.
* **Visualizes** the original, noisy, and retrieved patterns.
* Calculates **retrieval accuracy** across all digits.

---

## ⚙️ Requirements

* **Python 3.x**
* Libraries:

  * `torch` (PyTorch)
  * `torchvision` (for MNIST dataset)
  * `numpy`
  * `matplotlib`

Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib
```

---

## 📂 Files

* **`hopfield_mnist.py`** → Main script containing the Hopfield network implementation and evaluation.

---

## 🔎 How It Works

### 1. Data Preparation

* Loads the **MNIST dataset** and selects **50 samples per digit**.
* **Binarizes images** to ±1 (`pixels > 127 → 1`, else `-1`).
* Creates **prototype patterns** for each digit by summing and thresholding samples.

### 2. Hopfield Network Setup

* Constructs the **weight matrix** using the **outer product** of the 10 prototype patterns.
* **Normalizes** the weights and sets diagonal elements to **zero**.

### 3. Pattern Retrieval

* For each digit:

  * Distorts the prototype by flipping **20% of pixels**.
  * Updates the distorted pattern using Hopfield dynamics (**500 iterations**).
  * Predicts the digit by comparing the retrieved pattern to stored prototypes (via **dot product similarity**).

### 4. Evaluation & Visualization

* Displays **Original**, **Noisy**, and **Retrieved** patterns for each digit.
* Reports **accuracy**: how many retrieved patterns match the true digit.

---

## ▶️ Usage

1. Ensure dependencies are installed.
2. Run the script:

   ```bash
   python hopfield_mnist.py
   ```
3. The script will:

   * Download MNIST (if not already available).
   * Generate plots for each digit (**original, noisy, retrieved**).
   * Print network accuracy, e.g.:

     ```
     Accuracy: 8/10 digits (80%)
     ```

---

## 📊 Output

For each digit (**0–9**), the script produces:

* **Original prototype**
* **Noisy version** (20% pixels flipped)
* **Retrieved pattern** after Hopfield updates
* Title showing **True vs Predicted digit**

At the end:

* **Accuracy score** across all digits.

---

## 📝 Notes

* The Hopfield network retrieves patterns **close to the original**, not necessarily exact.
* The network’s **capacity is limited**, but with 10 digits it typically performs well.
* You can experiment by adjusting:

  * `samples_per_digit` → number of samples used per digit
  * `flip_fraction` → noise level
  * `iters` → update iterations

---

## ⚠️ Limitations

* Retrieval may not always yield the exact original due to **iterative dynamics** and **noise**.
* Performance decreases with **higher noise** or **more stored patterns**.



