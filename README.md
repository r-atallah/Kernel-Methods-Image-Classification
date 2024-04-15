# Kernel Methods for Machine Learning: Kaggle Challenge

This project was completed as part of the Kernel Methods for Machine Learning course in the MVA master, during the academic year 2023-2024, as a Kaggle challenge.

## Overview

In this project, we implemented kernel PCA, various kernel methods and feature extraction techniques from scratch. Use of external machine learning libraries for classifiers, such as sickit-learn, libsvm, pytorch is forbidden. Specifically, we implemented the following:

- **Kernel Ridge Classifier (KRC):** A classifier based on kernel ridge regression, implemented from scratch.
- **Kernel SVM with SMO Optimization:** A support vector machine (SVM) classifier with [Sequential Minimal Optimization (SMO)](https://www.microsoft.com/en-us/research/uploads/prod/1998/04/sequential-minimal-optimization.pdf) optimization, implemented from scratch.
- **Feature Extraction:**
  - [Histogram of Oriented Gradients (HOG)]()
  - [Scale Invariant Feature Transform (SIFT)](https://www.researchgate.net/publication/235355151_Scale_Invariant_Feature_Transform)
  - [Kernel Descriptors](https://rse-lab.cs.washington.edu/postscripts/kdes-nips-10.pdf)
  - [Fisher Vector](https://inria.hal.science/hal-00830491v2/document)

## How to Run

To launch the project and generate a submission file, use the following command:
``` 
python src/start.py
```
