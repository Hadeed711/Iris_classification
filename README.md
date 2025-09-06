# Iris Flower Classification — Mini Project

This project is a simple machine learning classification task using the classic **Iris dataset** from scikit-learn. The goal is to classify iris flowers into three species (*setosa*, *versicolor*, *virginica*) based on four measurements (sepal length, sepal width, petal length, petal width).

---

## Project Structure

```
├── iris_classification.ipynb   # Jupyter Notebook with code & analysis
├── iris_best_model.joblib      # Saved trained model (pipeline)
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## Dataset

* **Source:** `sklearn.datasets.load_iris`
* **Samples:** 150
* **Classes:** 3 (Setosa, Versicolor, Virginica)
* **Features:**

  * Sepal length (cm)
  * Sepal width (cm)
  * Petal length (cm)
  * Petal width (cm)

---

## Methodology

1. **Data Loading & EDA**

   * Converted dataset into pandas DataFrame.
   * Visualized distributions, pairplots, and correlation matrix.

2. **Preprocessing**

   * Train-test split (80/20, stratified).
   * StandardScaler included in ML pipelines.

3. **Model Comparison**

   * Logistic Regression
   * K-Nearest Neighbors
   * Decision Tree
   * Random Forest
   * Support Vector Machine (SVM)
   * Gaussian Naive Bayes

   → Compared using 5-fold cross-validation and test accuracy.

4. **Hyperparameter Tuning**

   * Tuned Random Forest and SVM using GridSearchCV.

5. **Final Model**

   * Selected the best-performing tuned model.
   * Evaluated on hold-out test set with classification report and confusion matrix.

6. **Model Saving**

   * Best model saved using `joblib`.

---

## Results

* All models performed well (>90% accuracy).
* SVM and Random Forest achieved the highest scores.
* Final tuned model reached \~97–100% accuracy on test set.

### Confusion Matrix Example

```
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
```

---

## How to Run

1. Clone this repository:

   ```bash
   git clone <repo_url>
   cd iris-classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook iris_classification.ipynb
   ```

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Future Work

* Try additional algorithms (XGBoost, LightGBM).
* Add model explainability (SHAP, permutation importance).
* Deploy model as a Flask/Streamlit web app.

---

## Author

**Hadeed Ahmad**
Student at University of Agriculture — Interested in Data Science, ML, and AI.
