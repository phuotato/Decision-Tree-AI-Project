# Decision Tree Classifier on Real-World Datasets

## Overview

This project implements decision tree classifiers using scikit-learn on three real-world datasets:  
- **Breast Cancer Wisconsin (Diagnostic) Dataset**  
- **Wine Quality Dataset**  
- **Anemia Types Classification Dataset**

The goal of the project is to classify the data based on decision tree algorithms, visualize the trees, evaluate their performance, and investigate how the depth of a decision tree influences accuracy.

## Datasets

### 1. UCI Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Link**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
- **Description**: The dataset contains 569 samples of cell features from breast cancer biopsies. Each sample is labeled as either benign (B) or malignant (M). This is a binary classification task.

### 2. UCI Wine Quality Dataset  
- **Link**: [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)  
- **Description**: The dataset consists of 4898 wine samples classified into 11 quality categories (0 to 10). The classification is based on physicochemical properties like alcohol content, acidity, etc. This is a multi-class classification task.

### 3. Anemia Types Classification Dataset  
- **Link**: https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification
- **Description**: 

## Project Specifications

### 1. Dataset Preparation  
The dataset will be preprocessed by splitting it into training and testing sets using the following proportions: 40/60, 60/40, 80/20, and 90/10. The data will be shuffled and split in a stratified fashion to ensure that class distributions are maintained. For each split, both features and labels will be separated into distinct training and testing subsets.

Additionally, the class distributions for the original, training, and testing datasets will be visualized to ensure proper data splitting.

### 2. Decision Tree Classifiers  
The main task is to train decision tree classifiers using the training data for each split and then evaluate the classifiers on the testing data. The following steps will be conducted:
- **Model Training**: A decision tree classifier from `sklearn.tree.DecisionTreeClassifier` will be used, with information gain (entropy) as the criterion.
- **Model Visualization**: After training, the decision tree will be visualized using Graphviz to show the tree's structure and decision boundaries.

### 3. Model Evaluation  
For each trained classifier, the following evaluations will be performed:
- **Classification Report**: Generated using `classification_report` to display precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Generated using `confusion_matrix` to visualize the classification results.
- **Interpretation**: Insights will be provided based on the classification report and confusion matrix, including potential improvements.

### 4. Depth vs Accuracy  
The effect of the maximum depth (`max_depth`) of the decision tree on classification accuracy will be investigated using the 80/20 training-test split. The classifier will be trained for various values of `max_depth` (None, 2, 3, 4, 5, 6, 7), and the following will be reported:
- **Visualization**: The decision tree will be visualized for each `max_depth` value.
- **Accuracy**: Accuracy scores will be provided for each `max_depth` setting.
- **Analysis**: A chart will be created to visualize how the depth affects the accuracy and to provide insights into the results.

## Files in This Repository

- `decision_tree_classifier1.ipynb`, `decision_tree_classifier2.ipynb`, `decision_tree_classifier3.ipynb`: Jupyter notebook containing the main code for data preprocessing, training decision tree classifiers, and evaluating their performance.
- `data/`: Folder containing the datasets (in `.csv` or `.xlsx` format).
- `plots/`: Folder containing the visualizations of decision trees and evaluation charts.
- `requirements.txt`: List of Python packages required to run the project.

## Requirements

To run this project, you need to have the following Python libraries installed:

- scikit-learn
- pandas
- matplotlib
- numpy
- graphviz
- seaborn

You can install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/phuotato/Decision-Tree-AI-Project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd decision-tree-project
   ```

3. Open the Jupyter notebook:

   ```bash
   jupyter notebook decision_tree_classifier.ipynb
   ```

4. Follow the steps in the notebook to execute the code and visualize the results.

## Insights & Conclusion

The results of this project will provide insights into the performance of decision tree classifiers on different datasets, how different training-test splits affect the results, and how varying the tree depth can impact the accuracy of the model. The project will also explore the practical considerations of decision tree models, including overfitting and underfitting, and offer recommendations for model improvements.
