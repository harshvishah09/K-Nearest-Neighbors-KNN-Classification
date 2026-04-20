# K-Nearest-Neighbors-KNN-Classification
# 🌸 Iris Classification using K-Nearest Neighbors (KNN)

## 📌 Project Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm to classify iris flowers into different species based on their features.
It includes model training, hyperparameter tuning (K value), evaluation, and visualization of decision boundaries.

---

## 📊 Dataset

* **Name:** Iris Dataset
* **Classes:** Setosa, Versicolor, Virginica
* **Features:**

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width

---

## 🚀 Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## ⚙️ Steps Performed

### 1. Data Preprocessing

* Loaded dataset using Pandas
* Removed unnecessary `Id` column
* Split data into features (`X`) and target (`y`)

---

### 2. Feature Scaling

* Applied **StandardScaler**
* Normalized data to improve KNN performance

---

### 3. Train-Test Split

* Split dataset into 80% training and 20% testing

---

### 4. Model Training (KNN)

* Used `KNeighborsClassifier`
* Tested multiple values of **K (1–10)**
* Selected the best K based on accuracy

---

### 5. Model Evaluation 📊

* Accuracy Score
* Confusion Matrix
* Visualization using `ConfusionMatrixDisplay`

---

### 6. K Value Optimization

* Plotted **K vs Accuracy graph**
* Selected optimal K for best performance

---

### 7. Decision Boundary Visualization

* Visualized classification regions using first two features
* Used mesh grid to plot decision boundaries

---

## 📈 Results

* Best K value selected automatically
* Model achieved high accuracy on test data
* Clear separation between classes observed in visualization

---

## 📂 Project Structure

```
📁 Iris-KNN-Project
│── iris_knn.py
│── Iris.csv
│── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/iris-knn-project.git
```

2. Navigate to the folder:

```
cd iris-knn-project
```

3. Run the script:

```
python iris_knn.py
```

---

## 🧠 Key Concepts

* **KNN (K-Nearest Neighbors):**
  A distance-based algorithm that classifies data based on nearest neighbors

* **Feature Scaling:**
  Important because KNN relies on distance calculations

* **Overfitting vs Underfitting:**
  Small K → overfitting
  Large K → underfitting

---

## ❓ Interview Questions

### 1. What is KNN?

A supervised learning algorithm that classifies data based on nearest neighbors.

### 2. Why is scaling important in KNN?

Because KNN uses distance metrics, and features must be on the same scale.

### 3. How do you choose K?

By testing multiple values and selecting the one with highest accuracy.

### 4. What is a decision boundary?

A region that separates different classes in the feature space.

---

## ✅ Conclusion

KNN is a simple yet powerful algorithm for classification tasks.
Proper feature scaling and optimal selection of K significantly improve performance.

---

## 📌 Author

harshvi shah
Your Name
