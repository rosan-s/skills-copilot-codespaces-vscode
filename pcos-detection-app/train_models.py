"""
PCOS Detection - Model Training and Comparison
This script trains 5 different ML/DL models and compares their performance
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class PCOSModelTrainer:
    def __init__(self, data_path='data/pcos_dataset.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.detailed_results = {}  # Store detailed analysis for each model
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('PCOS', axis=1)
        y = df['PCOS']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(self.feature_names)}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.detailed_results['Logistic Regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'strengths': [
                'Fast training and prediction',
                'Good interpretability',
                'Works well with linearly separable data',
                'Low computational cost'
            ],
            'weaknesses': [
                'Assumes linear relationship',
                'May underperform with complex patterns',
                'Sensitive to feature scaling'
            ]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        joblib.dump(model, 'models/logistic_regression.pkl')
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.detailed_results['Random Forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'strengths': [
                'Handles non-linear relationships well',
                'Resistant to overfitting',
                'Provides feature importance',
                'Works with imbalanced data'
            ],
            'weaknesses': [
                'Can be slow with large datasets',
                'Less interpretable than single trees',
                'Memory intensive'
            ]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        joblib.dump(model, 'models/random_forest.pkl')
        
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("\n" + "="*50)
        print("Training Support Vector Machine...")
        print("="*50)
        
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.models['SVM'] = model
        self.results['SVM'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.detailed_results['SVM'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'strengths': [
                'Effective in high-dimensional spaces',
                'Good generalization with RBF kernel',
                'Memory efficient',
                'Works well with clear margin of separation'
            ],
            'weaknesses': [
                'Slower with large datasets',
                'Sensitive to parameter tuning',
                'Less interpretable',
                'Requires feature scaling'
            ]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        joblib.dump(model, 'models/svm.pkl')
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.detailed_results['XGBoost'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'strengths': [
                'Excellent performance on tabular data',
                'Built-in regularization prevents overfitting',
                'Handles missing values automatically',
                'Fast training with parallelization',
                'Feature importance analysis'
            ],
            'weaknesses': [
                'Requires careful hyperparameter tuning',
                'Can overfit on small datasets',
                'Less interpretable than simple models'
            ]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        joblib.dump(model, 'models/xgboost.pkl')
        
    def train_deep_learning(self):
        """Train Deep Neural Network model"""
        print("\n" + "="*50)
        print("Training Deep Neural Network...")
        print("="*50)
        
        # Build the model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        y_pred_prob = model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        self.models['Deep Neural Network'] = model
        self.results['Deep Neural Network'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': accuracy,  # DNN doesn't use traditional CV
            'cv_std': 0.0
        }
        
        self.detailed_results['Deep Neural Network'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': accuracy,
            'cv_std': 0.0,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, y_pred),
            'strengths': [
                'Learns complex non-linear patterns',
                'Highly flexible architecture',
                'Scales well with large datasets',
                'Can improve with more data',
                'End-to-end learning'
            ],
            'weaknesses': [
                'Requires large amounts of data',
                'Computationally expensive',
                'Black box - less interpretable',
                'Prone to overfitting on small datasets',
                'Requires careful tuning'
            ]
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        model.save('models/deep_neural_network.h5')
        
    def compare_models(self):
        """Compare all models and generate visualizations"""
        print("\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        print("\nRanking by Accuracy:")
        print("-" * 80)
        for i, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"{i}. {model_name:25s} - Accuracy: {row['accuracy']:.4f} | "
                  f"Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f} | "
                  f"F1-Score: {row['f1_score']:.4f}")
        
        # Save comparison to CSV
        comparison_df.to_csv('models/model_comparison.csv')
        
        # Save detailed results
        import json
        with open('models/detailed_analysis.json', 'w') as f:
            json.dump(self.detailed_results, f, indent=2)
        
        # Create visualization
        self.plot_comparison(comparison_df)
        
        # Find best model and explain why
        best_model = comparison_df.index[0]
        second_best = comparison_df.index[1] if len(comparison_df) > 1 else None
        
        print(f"\n{'='*50}")
        print(f"RECOMMENDED MODEL: {best_model}")
        print(f"{'='*50}")
        print(f"Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
        print(f"Precision: {comparison_df.loc[best_model, 'precision']:.4f}")
        print(f"Recall: {comparison_df.loc[best_model, 'recall']:.4f}")
        print(f"F1-Score: {comparison_df.loc[best_model, 'f1_score']:.4f}")
        
        if second_best:
            acc_diff = (comparison_df.loc[best_model, 'accuracy'] - comparison_df.loc[second_best, 'accuracy']) * 100
            print(f"\nOutperforms {second_best} by {acc_diff:.2f}% in accuracy")
        
        print("\n" + "="*50)
        print("INDIVIDUAL MODEL ANALYSIS")
        print("="*50)
        for model_name in comparison_df.index[:3]:  # Top 3 models
            details = self.detailed_results.get(model_name, {})
            print(f"\n{model_name}:")
            print(f"  Strengths:")
            for strength in details.get('strengths', []):
                print(f"    ✓ {strength}")
            print(f"  Weaknesses:")
            for weakness in details.get('weaknesses', []):
                print(f"    ⚠ {weakness}")
        
        return comparison_df
        
    def plot_comparison(self, comparison_df):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        axes[0, 0].barh(comparison_df.index, comparison_df['accuracy'], color='skyblue')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_xlim([0, 1])
        for i, v in enumerate(comparison_df['accuracy']):
            axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # Precision comparison
        axes[0, 1].barh(comparison_df.index, comparison_df['precision'], color='lightcoral')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_title('Model Precision Comparison')
        axes[0, 1].set_xlim([0, 1])
        for i, v in enumerate(comparison_df['precision']):
            axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # Recall comparison
        axes[1, 0].barh(comparison_df.index, comparison_df['recall'], color='lightgreen')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Model Recall Comparison')
        axes[1, 0].set_xlim([0, 1])
        for i, v in enumerate(comparison_df['recall']):
            axes[1, 0].text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # F1-Score comparison
        axes[1, 1].barh(comparison_df.index, comparison_df['f1_score'], color='plum')
        axes[1, 1].set_xlabel('F1-Score')
        axes[1, 1].set_title('Model F1-Score Comparison')
        axes[1, 1].set_xlim([0, 1])
        for i, v in enumerate(comparison_df['f1_score']):
            axes[1, 1].text(v + 0.01, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved as 'models/model_comparison.png'")
        
    def train_all_models(self):
        """Train all models"""
        self.load_and_prepare_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_xgboost()
        self.train_deep_learning()
        comparison_df = self.compare_models()
        return comparison_df

if __name__ == "__main__":
    print("="*50)
    print("PCOS DETECTION - MODEL TRAINING")
    print("="*50)
    
    trainer = PCOSModelTrainer()
    results = trainer.train_all_models()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)
