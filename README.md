#Credit Card Fraud Detection

#Project Overview:
 Credit card fraud is a critical issue in the financial sector due to highly imbalanced transaction data and the high cost of false negatives.
 This project builds a machine learning pipeline to detect fraudulent credit card transactions using ensemble models and precision-recall–focused evaluation.
 The project emphasizes real-world ML practices such as handling imbalanced data, tracking experiments with MLflow, and using appropriate metrics beyond accuracy.

#Objectives:
 Detect fraudulent transactions with high recall and precision
 Handle class imbalance effectively
 Evaluate models using precision-recall curve, average precision score, and confusion matrix
 Track experiments using MLflow

#Technologies & Tools Used:
 Python
 Pandas, NumPy – data processing
 Scikit-learn – model training & evaluation
 imbalanced-learn – handling imbalanced datasets (SMOTE / undersampling)
 RandomForestClassifier / XGBoost
 MLflow – experiment tracking
 Matplotlib / Seaborn – visualization
 VS Code – development environment

#Dataset:
  Credit card transaction dataset

  Highly imbalanced:
  Legitimate transactions ≫ Fraudulent transactions
  Features are anonymized for privacy
  ~Dataset is not included in the repository due to size/privacy constraints.

#Project Structure
 credit_card_fraud_detection/
 │── data/                 
 │── notebooks/           
 │── src/                  
 │   ├── preprocess.py
 │   ├── train.py
 │   ├── evaluate.py
 │── models/               
 │── mlruns/               
 │── requirements.txt
 │── README.md

#Workflow:
 Data loading and preprocessing
 Handling class imbalance (SMOTE / sampling)
 Model training (Random Forest / XGBoost)

#Model evaluation using:
 Precision-Recall Curve
 Average Precision Score
 Confusion Matrix
 Experiment tracking with MLflow

#Model Evaluation Metrics:
  Due to extreme class imbalance, accuracy is not used.

  Key metrics:
  Precision-Recall Curve
  Average Precision Score
  Confusion Matrix

#MLflow Experiment Tracking:

  MLflow is used to:
  Track model parameters
  Log metrics
  Compare different model runs
  Store trained models

  To launch MLflow UI:
  mlflow ui


#How to Run the Project:

Clone the repository
```bash
git clone https://github.com/AkashT132006/credit_card_fraud_detection.git
cd credit_card_fraud_detection

Install dependencies:
pip install -r requirements.txt


Train the model:
python src/train.py

Evaluate the model:
python src/evaluate.py


#Key Learnings:
- Importance of precision-recall metrics for imbalanced datasets
- Practical techniques to handle class imbalance (SMOTE, sampling)
- End-to-end ML pipeline design
- Experiment tracking and comparison using MLflow



##Author##

**Akash T**  
Aspiring Machine Learning Engineer  
India


