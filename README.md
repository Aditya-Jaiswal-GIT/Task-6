# 🪀 Heart Disease Prediction using K-Nearest Neighbors (KNN)

This project applies the **K-Nearest Neighbors (KNN)** algorithm to predict heart disease presence from a dataset of patient health features. It includes model training, evaluation with different `k` values, and a 2D visualization using **PCA**.

---

## 📂 Dataset

* **Source:** `heart.csv`
* **Target Column:** `target` (1 = presence of heart disease, 0 = absence)
* **Features:** 13 numerical features including age, cholesterol, maximum heart rate, etc.

---

## 📊 Libraries Used

* `pandas`
* `numpy`
* `matplotlib`
* `sklearn` (`model_selection`, `preprocessing`, `neighbors`, `metrics`, `decomposition`)

---

## ⚙️ Workflow

1. **Load Data**
   Read and process the dataset using Pandas.

2. **Preprocessing**

   * Normalize features using `StandardScaler`.
   * Split into train and test sets (80/20).

3. **Model Training & Evaluation**

   * Trained KNN with `k = 3`, `5`, and `7`.
   * Evaluated using:

     * Accuracy
     * Confusion Matrix
     * Classification Report

4. **Visualization**

   * Reduced feature dimensions using **PCA** to 2D.
   * Visualized decision boundaries using a meshgrid and color maps.

---

## 📈 Output Example

Model accuracy and metrics are printed for each value of `k`. A decision boundary plot shows KNN classification in PCA-reduced space.

---

## 🧠 How to Run

1. Ensure you have the required libraries:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

2. Place `heart.csv` in the working directory.

3. Run the script:

   ```bash
   python knn_heart.py
   ```

---

## 📌 Notes

* PCA is used **only for visualization**, not for training.
* Make sure the dataset is clean and contains no missing values.
