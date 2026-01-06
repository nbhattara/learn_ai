Day 4 – Machine Learning Concepts and Practice
1. Recap of Previous Days

Day 1: Basics of AI & ML, types of ML (Supervised, Unsupervised, Reinforcement)

Day 2: ML workflow: Data Collection → Preprocessing → Model → Evaluation → Deployment

Day 3: Supervised Learning algorithms: Linear Regression, Logistic Regression, Decision Trees, Naive Bayes

Today we focus more on model evaluation, feature engineering, and some advanced algorithms.

2. Feature Engineering

Feature engineering is the process of transforming raw data into meaningful features that improve model performance.

Key Steps:

Handling Missing Values

Mean/Median/Mode imputation

Drop missing rows (if small portion)

Use prediction models for imputation

Encoding Categorical Variables

Label Encoding → Assign integers to categories

One-Hot Encoding → Convert categories into binary columns

Scaling Features

Standardization (Z-score) → (x - mean)/std

Normalization (Min-Max) → (x - min)/(max - min)

Feature Selection

Remove irrelevant features

Use techniques like Correlation Matrix, PCA, or Recursive Feature Elimination (RFE)

3. Model Evaluation Metrics

Knowing how to evaluate your model is crucial.

For Regression

Mean Absolute Error (MAE): Average of absolute errors

Mean Squared Error (MSE): Average of squared errors (penalizes large errors)

R² Score: How much variance is explained by the model

For Classification

Accuracy: (TP + TN) / (TP + TN + FP + FN)

Precision: TP / (TP + FP) → How many predicted positive are actually positive

Recall (Sensitivity): TP / (TP + FN) → How many actual positives are captured

F1-Score: Harmonic mean of Precision & Recall

Confusion Matrix: Visual representation of TP, TN, FP, FN

4. Cross-Validation

Instead of a single train-test split, cross-validation ensures your model generalizes well.

K-Fold CV: Split data into k parts, train on k-1 and test on 1, repeat k times

Stratified K-Fold: Keeps the proportion of classes same in each fold (for classification)

5. Advanced Algorithms

Some algorithms you should start exploring:

Support Vector Machines (SVM)

Finds the optimal hyperplane that separates classes

Works well in high-dimensional spaces

Random Forest

Ensemble of Decision Trees

Reduces overfitting of single trees

Good for both regression & classification

Gradient Boosting (XGBoost, LightGBM, CatBoost)

Sequentially builds trees to correct previous errors

Powerful for tabular datasets

6. Overfitting & Underfitting

Overfitting: Model performs well on training data but poorly on unseen data
Solutions: More data, regularization, pruning, dropout (for neural networks)

Underfitting: Model too simple to capture patterns
Solutions: More complex model, better features