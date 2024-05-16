# Mini-Scikit-Learn

Mini-Scikit-Learn is a lightweight machine learning library inspired by Scikit-Learn. This project aims to implement essential machine learning algorithms, preprocessing techniques, model evaluation methods, and utilities to provide a basic yet functional machine learning toolkit.

## Project Structure

The project is organized into several directories, each containing Python modules and Jupyter notebooks for different aspects of machine learning:

- **ensemble**: Contains implementations of various ensemble methods including Random Forest.
- **metrics**: Includes modules for evaluating model performance such as accuracy, precision, recall, F1 score, and confusion matrix.
- **model_selection**: Features tools for model selection and hyperparameter tuning, including train-test split and GridSearchCV.
- **neural_networks**: Dedicated to basic neural network architectures.
- **preprocessing**: Holds preprocessing utilities like data scaling and encoding.
- **supervised_learning**: Contains implementations of supervised learning algorithms like Logistic Regression, KNN, Decision Trees, etc.
- **utilities**: Utility functions and classes used across the project.

Each directory contains Jupyter notebooks that demonstrate the testing of the respective modules implemented in the project.

### Notebooks

- **ClassificationMetricsTest.ipynb**: Tests and comparisons of classification metrics.
- **DecisionTreeClassifier.ipynb**: Demonstrations of the Decision Tree classifier.
- **DecisionTreeRegressor.ipynb**: Demonstrations of the Decision Tree regressor.
- **GridSearchCVTest.ipynb**: Usage examples for GridSearchCV.
- Other notebooks follow a similar naming convention, each focusing on different components of the library.

## Installation

To use Mini-Scikit-Learn, clone this repository to your local machine. Ensure that you have Python installed, along with the necessary libraries.

```bash
git clone https://github.com/Basma-Arnaoui/Mini-Scikit-Learn.git
cd Mini-Scikit-Learn
```

## Usage

To use the components of Mini-Scikit-Learn, you can import the required modules into your Python scripts or Jupyter notebooks. For example:

```python
from supervised_learning.classification import LogisticRegression
from model_selection import GridSearchCV

# Your code to use these components goes here
```