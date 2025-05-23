# Titanic Survival Classification 🚢

A machine learning project that predicts passenger survival on the Titanic using the Titanic dataset. The project involves data preprocessing, feature engineering, and hyperparameter-tuned K-Nearest Neighbors (KNN) classification.

## Project Structure

titanic/
├── main.py
├── Titanic-Dataset.csv
├── README.md
└── confusion_matrix.png 


## Overview

This project uses a supervised machine learning approach to classify passengers based on their survival status (`Survived` = 0 or 1). The key steps involved include:

- Handling missing values
- Encoding categorical data
- Feature engineering (e.g., `FamilySize`, `IsAlone`, `FareBin`, `AgeBin`)
- Data normalization using MinMaxScaler
- Hyperparameter tuning with GridSearchCV
- Classification using K-Nearest Neighbors

## How to Run

1. Ensure you have Python 3 and pip installed.
2. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

3. Run the script:

```bash
python main.py


## Model Performance

- **Classifier Used:** K-Nearest Neighbors (tuned with GridSearchCV)
- **Accuracy:** **80.72%**

### Classification Report

| Class                 | Precision | Recall | F1 Score | Support |
|-----------------------|-----------|--------|----------|---------|
| 0 (Did Not Survive)   | 0.81      | 0.88   | 0.85     | 134     |
| 1 (Survived)          | 0.79      | 0.70   | 0.74     | 89      |

**Macro Avg F1-Score:** 0.79  
**Weighted Avg F1-Score:** 0.80  

### 📌 Confusion Matrix

|                 | Predicted: 0 | Predicted: 1 |
|-----------------|--------------|--------------|
| **Actual: 0**   |     118      |      16      |
| **Actual: 1**   |      27      |      62      |

Or view it visually:

![Confusion Matrix](confusion_matrix.png)


## Notes
- The dataset is sourced from Kaggle Titanic Dataset.
- You can modify the model or preprocessing logic in ```main.py```.

## Author
Debb – @debb-major. Big shout out to 'Code with Josh' on YouTube


## License
This project is open-source and available under the MIT License.

