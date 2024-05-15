import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import argparse
from tqdm import tqdm
import os

def preprocess_data(data):
    data = np.array(data)

    # Debugging: Check for infinity and NaN before replacement
    print("Before replacement:")
    print(f"Contains infinity: {np.isinf(data).any()}")
    print(f"Contains NaN: {np.isnan(data).any()}")
    
    # Replace extreme large values with a large finite number
    max_finite_value = np.finfo(np.float64).max / 2  # Use half of max to prevent overflow
    data = np.clip(data, -max_finite_value, max_finite_value)
    
    # Replace infinity and NaN with zero
    data[np.isinf(data)] = 0
    data[np.isnan(data)] = 0
    
    # Debugging: Check for infinity and NaN after replacement
    print("After replacement:")
    print(f"Contains infinity: {np.isinf(data).any()}")
    print(f"Contains NaN: {np.isnan(data).any()}")

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Check for NaN values after scaling
    if np.isnan(data_scaled).any():
        print("Data contains NaN values after scaling.")
        # Handle NaNs after scaling if necessary
        data_scaled = np.nan_to_num(data_scaled)  # Replace NaNs with zero

    return data_scaled, scaler

def train_svm(X_train, y_train):
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)
    return svm_model

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    # Use GridSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    total_params = np.prod([len(param_grid[key]) for key in param_grid])
    print(f"Total Random Forest parameter combinations: {total_params}")

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    
    with tqdm(total=total_params, desc="Random Forest GridSearch") as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(total_params)
    
    return grid_search.best_estimator_, grid_search.cv_results_

def train_gradient_boosting(X_train, y_train):
    gb_model = GradientBoostingClassifier(random_state=42)
    # Use GridSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    total_params = np.prod([len(param_grid[key]) for key in param_grid])
    print(f"Total Gradient Boosting parameter combinations: {total_params}")

    grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)

    with tqdm(total=total_params, desc="Gradient Boosting GridSearch") as pbar:
        grid_search.fit(X_train, y_train)
        pbar.update(total_params)
    
    return grid_search.best_estimator_, grid_search.cv_results_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    return report, accuracy

def save_model(model, scaler, model_path='model.joblib', scaler_path='scaler.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path} and scaler saved to {scaler_path}")

def load_model(model_path='model.joblib', scaler_path='scaler.joblib'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def save_grid_search_results(cv_results, model_name):
    results_df = pd.DataFrame(cv_results)
    results_df['model'] = model_name
    results_file = 'grid_search_results.csv'
    
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)
    
    print(f"Grid search results saved to {results_file}")

def main(model_type=None):
    # Load the CSI data and labels from numpy files
    data = np.load('csi_data.npy')
    labels = np.load('csi_labels.npy')

    data_scaled, scaler = preprocess_data(data)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.3, random_state=42)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    results = []

    if model_type == 'svm':
        # Train the SVM model
        model = train_svm(X_train_resampled, y_train_resampled)
        model_path = 'svm_model.joblib'
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'SVM', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path)
    elif model_type == 'rf':
        # Train the Random Forest model
        model, cv_results = train_random_forest(X_train_resampled, y_train_resampled)
        model_path = 'rf_model.joblib'
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'Random Forest', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path)
        save_grid_search_results(cv_results, 'Random Forest')
    elif model_type == 'gb':
        # Train the Gradient Boosting model
        model, cv_results = train_gradient_boosting(X_train_resampled, y_train_resampled)
        model_path = 'gb_model.joblib'
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'Gradient Boosting', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path)
        save_grid_search_results(cv_results, 'Gradient Boosting')
    else:
        # Train and evaluate all models
        print("Training SVM...")
        model = train_svm(X_train_resampled, y_train_resampled)
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'SVM', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path='svm_model.joblib')

        print("Training Random Forest...")
        model, cv_results = train_random_forest(X_train_resampled, y_train_resampled)
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'Random Forest', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path='rf_model.joblib')
        save_grid_search_results(cv_results, 'Random Forest')

        print("Training Gradient Boosting...")
        model, cv_results = train_gradient_boosting(X_train_resampled, y_train_resampled)
        report, accuracy = evaluate_model(model, X_test, y_test)
        results.append({'Model': 'Gradient Boosting', 'Accuracy': accuracy, 'Parameters': model.get_params()})
        save_model(model, scaler, model_path='gb_model.joblib')
        save_grid_search_results(cv_results, 'Gradient Boosting')

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_file = 'model_results.csv'
    
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument('--model', type=str, choices=['svm', 'rf', 'gb'], help='Model to train (svm, rf, gb)')
    args = parser.parse_args()
    
    main(model_type=args.model)
