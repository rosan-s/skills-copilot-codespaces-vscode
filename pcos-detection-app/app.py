"""
PCOS Detection Web Application - Flask Backend
"""

# Suppress TensorFlow warnings for faster startup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import json
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⊘ TensorFlow not available - using sklearn models only")
from werkzeug.utils import secure_filename
import io
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store uploaded dataset analysis
uploaded_dataset_analysis = {}

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Load models and scaler
models = {}
model_info = {}
detailed_analysis = {}

def load_all_models():
    """Load all trained models"""
    global models, model_info, detailed_analysis
    
    try:
        print("Loading models from disk...")
        
        # Check if models directory exists
        if not os.path.exists('models'):
            print("Models directory not found! Creating...")
            os.makedirs('models', exist_ok=True)
            print("Please run train_models.py to train the models first.")
            return False
        
        # Load each model with error handling
        try:
            models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
            print("✓ Loaded Logistic Regression")
        except Exception as e:
            print(f"✗ Failed to load Logistic Regression: {e}")
            
        try:
            models['Random Forest'] = joblib.load('models/random_forest.pkl')
            print("✓ Loaded Random Forest")
        except Exception as e:
            print(f"✗ Failed to load Random Forest: {e}")
            
        try:
            models['SVM'] = joblib.load('models/svm.pkl')
            print("✓ Loaded SVM")
        except Exception as e:
            print(f"✗ Failed to load SVM: {e}")
            
        try:
            models['XGBoost'] = joblib.load('models/xgboost.pkl')
            print("✓ Loaded XGBoost")
        except Exception as e:
            print(f"✗ Failed to load XGBoost: {e}")
            
        # Skip Deep Neural Network on production for free tier compatibility
        if TENSORFLOW_AVAILABLE and os.path.exists('models/deep_neural_network.h5') and os.getenv('SKIP_DNN') != '1':
            try:
                models['Deep Neural Network'] = keras.models.load_model('models/deep_neural_network.h5')
                print("✓ Loaded Deep Neural Network")
            except Exception as e:
                print(f"✗ Failed to load Deep Neural Network (skipping): {e}")
        else:
            print("⊘ Skipping Deep Neural Network (free tier optimization)")
            
        try:
            models['scaler'] = joblib.load('models/scaler.pkl')
            print("✓ Loaded Scaler")
        except Exception as e:
            print(f"✗ Failed to load Scaler: {e}")
        
        # Load model comparison data
        if os.path.exists('models/model_comparison.csv'):
            comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
            model_info = comparison_df.to_dict('index')
        
        # Load detailed analysis
        if os.path.exists('models/detailed_analysis.json'):
            with open('models/detailed_analysis.json', 'r') as f:
                detailed_analysis = json.load(f)
        
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Feature names (must match training data)
FEATURE_NAMES = [
    'Age', 'BMI', 'Cycle_length', 'FSH', 'LH', 'TSH', 'AMH', 'Insulin',
    'Weight_gain', 'Hair_growth', 'Skin_darkening', 'Hair_loss', 
    'Pimples', 'Fast_food', 'Reg_Exercise'
]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get model comparison data"""
    try:
        if os.path.exists('models/model_comparison.csv'):
            comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
            models_data = []
            
            for idx, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
                model_details = detailed_analysis.get(model_name, {})
                models_data.append({
                    'rank': idx,
                    'name': model_name,
                    'accuracy': round(row['accuracy'] * 100, 2),
                    'precision': round(row['precision'] * 100, 2),
                    'recall': round(row['recall'] * 100, 2),
                    'f1_score': round(row['f1_score'] * 100, 2),
                    'strengths': model_details.get('strengths', []),
                    'weaknesses': model_details.get('weaknesses', []),
                    'confusion_matrix': model_details.get('confusion_matrix', [])
                })
            
            return jsonify({
                'success': True,
                'models': models_data,
                'best_model': models_data[0]['name'],
                'recommendation': f"{models_data[0]['name']} is recommended for PCOS detection based on highest accuracy and performance metrics."
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Models not trained yet. Please run train_models.py first.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def generate_model_graphs():
    """Generate performance graphs for all models"""
    try:
        if not os.path.exists('models/model_comparison.csv'):
            return None
        
        comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = comparison_df[metric] * 100
            bars = ax.bar(range(len(comparison_df)), values, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel(f'{name} (%)', fontweight='bold')
            ax.set_title(f'{name} Comparison', fontweight='bold', fontsize=12)
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Encode to base64
        graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return graph_base64
        
    except Exception as e:
        print(f"Error generating graphs: {e}")
        return None

def generate_confusion_matrix_graph(model_name):
    """Generate confusion matrix heatmap for a specific model"""
    try:
        if model_name not in detailed_analysis:
            return None
        
        cm = detailed_analysis[model_name].get('confusion_matrix', [])
        if not cm or len(cm) == 0:
            return None
        
        cm_array = np.array(cm)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'},
                   xticklabels=['No PCOS', 'PCOS'],
                   yticklabels=['No PCOS', 'PCOS'],
                   ax=ax, linewidths=2, linecolor='black')
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        
        # Add accuracy annotation
        total = cm_array.sum()
        correct = cm_array[0, 0] + cm_array[1, 1]
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%', 
                transform=ax.transAxes, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        plt.tight_layout()
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Encode to base64
        graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return graph_base64
        
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return None

@app.route('/api/model-graphs', methods=['GET'])
def get_model_graphs():
    """Get performance comparison graphs"""
    try:
        graph_data = generate_model_graphs()
        
        if graph_data:
            return jsonify({
                'success': True,
                'graph': graph_data
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Unable to generate graphs'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/confusion-matrix/<model_name>', methods=['GET'])
def get_confusion_matrix(model_name):
    """Get confusion matrix for a specific model"""
    try:
        graph_data = generate_confusion_matrix_graph(model_name)
        
        if graph_data:
            return jsonify({
                'success': True,
                'graph': graph_data,
                'model': model_name
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Unable to generate confusion matrix for {model_name}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using all models"""
    try:
        # Check if models are loaded
        if not models or len(models) == 0:
            return jsonify({
                'success': False,
                'message': 'Models not loaded. Server may be starting up. Please wait a moment and try again.'
            }), 503
        
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No input data provided'
            }), 400
        
        # Extract features in correct order
        features = []
        for feature in FEATURE_NAMES:
            if feature in data:
                features.append(float(data[feature]))
            else:
                return jsonify({
                    'success': False,
                    'message': f'Missing feature: {feature}'
                }), 400
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        if 'scaler' not in models:
            return jsonify({
                'success': False,
                'message': 'Scaler not loaded. Please train models first.'
            })
        
        features_scaled = models['scaler'].transform(features_array)
        
        # Get predictions from all models (prioritize fast sklearn models)
        predictions = {}
        
        # Process fast models first for better timeout handling
        fast_models_order = ['XGBoost', 'Random Forest', 'SVM', 'Logistic Regression', 'Deep Neural Network']
        
        for model_name in fast_models_order:
            if model_name not in models or model_name == 'scaler':
                continue
            
            model = models[model_name]
                
            try:
                if model_name == 'Deep Neural Network':
                    # DNN can be slow, skip on error to prevent timeout
                    try:
                        pred_prob = model.predict(features_scaled, verbose=0)[0][0]
                        pred = 1 if pred_prob > 0.5 else 0
                        confidence = pred_prob if pred == 1 else (1 - pred_prob)
                    except:
                        # Skip DNN if it times out or fails
                        continue
                else:
                    pred = model.predict(features_scaled)[0]
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features_scaled)[0]
                        confidence = pred_proba[pred]
                    else:
                        confidence = 0.95  # Default for models without probability
                
                predictions[model_name] = {
                    'prediction': int(pred),
                    'confidence': float(confidence) * 100,
                    'result': 'PCOS Detected' if pred == 1 else 'No PCOS'
                }
                
                # Add accuracy from training
                if model_name in model_info:
                    predictions[model_name]['accuracy'] = round(model_info[model_name]['accuracy'] * 100, 2)
                    
            except Exception as e:
                predictions[model_name] = {
                    'prediction': None,
                    'error': str(e)
                }
        
        # Determine consensus
        votes = [pred['prediction'] for pred in predictions.values() if 'prediction' in pred and pred['prediction'] is not None]
        consensus = 1 if sum(votes) > len(votes) / 2 else 0
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'consensus': {
                'prediction': int(consensus),
                'result': 'PCOS Detected' if consensus == 1 else 'No PCOS',
                'votes': f"{sum(votes)}/{len(votes)} models predict PCOS"
            },
            'input_data': data
        })
        
    except ValueError as ve:
        return jsonify({
            'success': False,
            'message': f'Invalid input data: {str(ve)}'
        }), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_loaded = len(models) > 0
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'models_loaded': list(models.keys())
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_dataset_statistics_graph(df):
    """Generate visualization of dataset statistics"""
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # 1. Class Distribution (if PCOS column exists)
        if 'PCOS' in df.columns:
            ax1 = fig.add_subplot(gs[0, 0])
            pcos_counts = df['PCOS'].value_counts()
            colors_pie = ['#10b981', '#ef4444']
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax1.pie(
                [pcos_counts.get(0, 0), pcos_counts.get(1, 0)],
                labels=['No PCOS', 'PCOS'],
                autopct='%1.1f%%',
                colors=colors_pie,
                explode=explode,
                shadow=True,
                startangle=90
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax1.set_title('Class Distribution', fontweight='bold', fontsize=12)
        
        # 2. Feature Count
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, f'{df.shape[1]}', 
                ha='center', va='center', fontsize=48, fontweight='bold', color='#667eea')
        ax2.text(0.5, 0.2, 'Total Features', 
                ha='center', va='center', fontsize=14, color='#666')
        ax2.text(0.5, 0.05, f'{df.shape[0]} samples', 
                ha='center', va='center', fontsize=12, color='#999')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_facecolor('#f5f7fa')
        
        # 3. Missing Values
        ax3 = fig.add_subplot(gs[1, 0])
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) > 0:
            missing_sorted = missing.sort_values(ascending=True)
            ax3.barh(range(len(missing_sorted)), missing_sorted.values, color='#ef5350', alpha=0.8)
            ax3.set_yticks(range(len(missing_sorted)))
            ax3.set_yticklabels(missing_sorted.index)
            ax3.set_xlabel('Count', fontweight='bold')
            ax3.set_title('Missing Values by Feature', fontweight='bold', fontsize=12)
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '✓ No Missing Values', 
                    ha='center', va='center', fontsize=16, fontweight='bold', color='#10b981')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_facecolor('#f0fdf4')
        
        # 4. Data Types Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        dtype_counts = df.dtypes.value_counts()
        colors_dtype = ['#667eea', '#764ba2', '#f093fb', '#4facfe'][:len(dtype_counts)]
        ax4.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.0f%%',
               colors=colors_dtype, shadow=True, startangle=45)
        ax4.set_title('Data Types Distribution', fontweight='bold', fontsize=12)
        
        # 5. Feature Statistics Summary
        ax5 = fig.add_subplot(gs[2, :])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Get basic stats for numeric columns
            stats_data = []
            for col in numeric_cols[:5]:  # Show first 5 numeric features
                stats_data.append({
                    'Feature': col,
                    'Mean': f"{df[col].mean():.2f}",
                    'Std': f"{df[col].std():.2f}",
                    'Min': f"{df[col].min():.2f}",
                    'Max': f"{df[col].max():.2f}"
                })
            
            # Create table
            table_data = []
            for stat in stats_data:
                table_data.append([stat['Feature'], stat['Mean'], stat['Std'], stat['Min'], stat['Max']])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Feature', 'Mean', 'Std Dev', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Color header
            for i in range(5):
                table[(0, i)].set_facecolor('#667eea')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stats_data) + 1):
                for j in range(5):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f5f7fa')
                    else:
                        table[(i, j)].set_facecolor('#ffffff')
            
            ax5.axis('off')
            ax5.set_title('Numeric Features Statistics (First 5)', fontweight='bold', fontsize=12, pad=20)
        
        plt.suptitle('Dataset Statistics Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        graph_base64 = base64.b64encode(buffer.getvalue()).decode()
        return graph_base64
        
    except Exception as e:
        print(f"Error generating dataset statistics graph: {e}")
        return None

def analyze_dataset(df):
    """Perform comprehensive analysis on uploaded dataset"""
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'basic_info': {},
        'statistics': {},
        'correlations': {},
        'class_distribution': {},
        'feature_analysis': {}
    }
    
    # Basic information
    analysis['basic_info'] = {
        'total_samples': int(df.shape[0]),
        'total_features': int(df.shape[1]),
        'feature_names': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    # Check if PCOS column exists
    if 'PCOS' in df.columns:
        pcos_counts = df['PCOS'].value_counts()
        analysis['class_distribution'] = {
            'PCOS_positive': int(pcos_counts.get(1, 0)),
            'PCOS_negative': int(pcos_counts.get(0, 0)),
            'PCOS_percentage': round((pcos_counts.get(1, 0) / len(df)) * 100, 2)
        }
        
        # Feature analysis by PCOS status
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'PCOS' in numeric_cols:
            numeric_cols.remove('PCOS')
        
        feature_stats = {}
        for col in numeric_cols:
            pcos_positive = df[df['PCOS'] == 1][col]
            pcos_negative = df[df['PCOS'] == 0][col]
            
            feature_stats[col] = {
                'pcos_positive_mean': round(float(pcos_positive.mean()), 2) if not pcos_positive.empty else 0,
                'pcos_negative_mean': round(float(pcos_negative.mean()), 2) if not pcos_negative.empty else 0,
                'overall_mean': round(float(df[col].mean()), 2),
                'overall_std': round(float(df[col].std()), 2),
                'min': round(float(df[col].min()), 2),
                'max': round(float(df[col].max()), 2)
            }
        
        analysis['feature_analysis'] = feature_stats
    
    # Statistical summary
    stats_df = df.describe()
    analysis['statistics'] = stats_df.to_dict()
    
    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        
        # Find top correlations with PCOS (if exists)
        if 'PCOS' in corr_matrix.columns:
            pcos_corr = corr_matrix['PCOS'].drop('PCOS').sort_values(ascending=False)
            analysis['correlations']['with_pcos'] = {
                k: round(float(v), 3) for k, v in pcos_corr.items()
            }
        
        # Full correlation matrix
        analysis['correlations']['matrix'] = {
            k: {k2: round(float(v2), 3) for k2, v2 in v.items()}
            for k, v in corr_matrix.to_dict().items()
        }
    
    return analysis

def predict_dataset_with_models(df):
    """Predict PCOS for entire dataset using all models"""
    try:
        # Check if required features exist
        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            return {
                'error': f'Missing required features: {", ".join(missing_features)}',
                'predictions': None
            }
        
        # Extract features
        X = df[FEATURE_NAMES].values
        
        # Scale features
        X_scaled = models['scaler'].transform(X)
        
        # Get predictions from all models
        model_predictions = {}
        
        for model_name, model in models.items():
            if model_name == 'scaler':
                continue
            
            try:
                if model_name == 'Deep Neural Network':
                    pred_probs = model.predict(X_scaled, verbose=0)
                    preds = (pred_probs > 0.5).astype(int).flatten()
                else:
                    preds = model.predict(X_scaled)
                
                model_predictions[model_name] = preds.tolist()
                
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                model_predictions[model_name] = None
        
        # Calculate accuracy if PCOS column exists
        model_accuracies = {}
        model_prediction_counts = {}
        
        if 'PCOS' in df.columns:
            y_true = df['PCOS'].values
            
            for model_name, preds in model_predictions.items():
                if preds is not None:
                    correct = sum(1 for i, pred in enumerate(preds) if pred == y_true[i])
                    accuracy = (correct / len(y_true)) * 100
                    model_accuracies[model_name] = round(accuracy, 2)
                    
                    # Count PCOS vs Non-PCOS predictions
                    pcos_count = sum(1 for p in preds if p == 1)
                    non_pcos_count = sum(1 for p in preds if p == 0)
                    model_prediction_counts[model_name] = {
                        'pcos': pcos_count,
                        'non_pcos': non_pcos_count,
                        'total': len(preds)
                    }
        else:
            # Even without ground truth, count predictions
            for model_name, preds in model_predictions.items():
                if preds is not None:
                    pcos_count = sum(1 for p in preds if p == 1)
                    non_pcos_count = sum(1 for p in preds if p == 0)
                    model_prediction_counts[model_name] = {
                        'pcos': pcos_count,
                        'non_pcos': non_pcos_count,
                        'total': len(preds)
                    }
        
        return {
            'predictions': model_predictions,
            'accuracies': model_accuracies,
            'prediction_counts': model_prediction_counts,
            'total_samples': len(df)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'predictions': None
        }

def generate_prediction_report_graph(df, prediction_results):
    """Generate visual report for dataset predictions"""
    try:
        if not prediction_results.get('accuracies'):
            return None
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Model Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, :])
        accuracies = prediction_results['accuracies']
        model_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = ax1.bar(range(len(model_names)), accuracy_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, accuracy_values)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax1.set_title('Model Prediction Accuracy on Uploaded Dataset', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Class Distribution in Dataset
        if 'PCOS' in df.columns:
            ax2 = fig.add_subplot(gs[1, 0])
            pcos_counts = df['PCOS'].value_counts()
            
            colors_pie = ['#10b981', '#ef4444']
            explode = (0.05, 0.05)
            
            ax2.pie([pcos_counts.get(0, 0), pcos_counts.get(1, 0)], 
                   labels=['No PCOS', 'PCOS'],
                   autopct='%1.1f%%',
                   colors=colors_pie,
                   explode=explode,
                   shadow=True,
                   startangle=90)
            ax2.set_title('Actual PCOS Distribution', fontweight='bold', fontsize=12)
        
        # 3. Prediction Distribution (Best Model)
        ax3 = fig.add_subplot(gs[1, 1])
        best_model = max(accuracies.items(), key=lambda x: x[1])[0]
        best_preds = prediction_results['predictions'][best_model]
        
        pred_counts = pd.Series(best_preds).value_counts()
        colors_pie = ['#10b981', '#ef4444']
        explode = (0.05, 0.05)
        
        ax3.pie([pred_counts.get(0, 0), pred_counts.get(1, 0)], 
               labels=['No PCOS', 'PCOS'],
               autopct='%1.1f%%',
               colors=colors_pie,
               explode=explode,
               shadow=True,
               startangle=90)
        ax3.set_title(f'Predicted Distribution ({best_model})', fontweight='bold', fontsize=12)
        
        # 4. Model Agreement Analysis
        ax4 = fig.add_subplot(gs[2, :])
        
        # Calculate consensus for each sample
        predictions_array = np.array([prediction_results['predictions'][m] for m in model_names if prediction_results['predictions'][m]])
        consensus = np.sum(predictions_array, axis=0)
        
        consensus_counts = pd.Series(consensus).value_counts().sort_index()
        
        ax4.bar(consensus_counts.index, consensus_counts.values, color='#667eea', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Number of Models Predicting PCOS', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
        ax4.set_title('Model Agreement Distribution', fontweight='bold', fontsize=12)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Total Samples: {prediction_results['total_samples']}\n"
        stats_text += f"Best Model: {best_model} ({accuracies[best_model]:.1f}%)\n"
        stats_text += f"Average Accuracy: {np.mean(list(accuracies.values())):.1f}%"
        
        plt.figtext(0.99, 0.01, stats_text, ha='right', va='bottom', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=10, family='monospace')
        
        plt.suptitle('Dataset Analysis Report - Model Predictions', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        # Encode to base64
        graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return graph_base64
        
    except Exception as e:
        print(f"Error generating prediction report: {e}")
        return None

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload and analyze dataset"""
    global uploaded_dataset_analysis
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Only CSV files are allowed.'
            }), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error reading CSV file: {str(e)}'
            }), 400
        
        # Validate dataset
        if df.empty:
            return jsonify({
                'success': False,
                'message': 'Dataset is empty'
            }), 400
        
        # Perform statistical analysis
        analysis = analyze_dataset(df)
        
        # Generate dataset statistics visualization
        dataset_graph = generate_dataset_statistics_graph(df)
        analysis['dataset_graph'] = dataset_graph
        
        # Predict with all models if features are available
        prediction_results = None
        prediction_graph = None
        
        has_required_features = all(f in df.columns for f in FEATURE_NAMES)
        
        if has_required_features:
            prediction_results = predict_dataset_with_models(df)
            
            if prediction_results.get('predictions') and not prediction_results.get('error'):
                # Generate visual report
                prediction_graph = generate_prediction_report_graph(df, prediction_results)
                
                # Add to analysis
                analysis['model_predictions'] = {
                    'accuracies': prediction_results.get('accuracies', {}),
                    'prediction_counts': prediction_results.get('prediction_counts', {}),
                    'total_samples': prediction_results.get('total_samples', 0),
                    'best_model': max(prediction_results.get('accuracies', {}).items(), 
                                    key=lambda x: x[1])[0] if prediction_results.get('accuracies') else None,
                    'has_visual_report': prediction_graph is not None
                }
        
        # Save analysis for later download
        uploaded_dataset_analysis = analysis
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        df.to_csv(filepath, index=False)
        
        analysis['saved_filename'] = saved_filename
        analysis['prediction_graph'] = prediction_graph
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded and analyzed successfully',
            'analysis': analysis,
            'has_predictions': prediction_results is not None and not prediction_results.get('error'),
            'prediction_error': prediction_results.get('error') if prediction_results else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing dataset: {str(e)}'
        }), 500

@app.route('/api/download-analysis', methods=['GET'])
def download_analysis():
    """Download analysis report"""
    try:
        format_type = request.args.get('format', 'json')
        
        if not uploaded_dataset_analysis:
            return jsonify({
                'success': False,
                'message': 'No dataset analysis available. Please upload a dataset first.'
            }), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            # Create JSON file
            json_str = json.dumps(uploaded_dataset_analysis, indent=2)
            buffer = io.BytesIO(json_str.encode())
            buffer.seek(0)
            
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'pcos_analysis_{timestamp}.json',
                mimetype='application/json'
            )
        
        elif format_type == 'csv':
            # Create comprehensive CSV report
            report_data = []
            
            # Basic info
            report_data.append(['PCOS Dataset Analysis Report'])
            report_data.append(['Generated:', uploaded_dataset_analysis.get('timestamp', '')])
            report_data.append([])
            
            # Basic information
            report_data.append(['Basic Information'])
            basic_info = uploaded_dataset_analysis.get('basic_info', {})
            for key, value in basic_info.items():
                if key not in ['feature_names', 'missing_values', 'data_types']:
                    report_data.append([key, value])
            report_data.append([])
            
            # Class distribution
            report_data.append(['Class Distribution'])
            class_dist = uploaded_dataset_analysis.get('class_distribution', {})
            for key, value in class_dist.items():
                report_data.append([key, value])
            report_data.append([])
            
            # Feature analysis
            report_data.append(['Feature Analysis'])
            report_data.append(['Feature', 'PCOS+ Mean', 'PCOS- Mean', 'Overall Mean', 'Std Dev', 'Min', 'Max'])
            feature_analysis = uploaded_dataset_analysis.get('feature_analysis', {})
            for feature, stats in feature_analysis.items():
                report_data.append([
                    feature,
                    stats.get('pcos_positive_mean', ''),
                    stats.get('pcos_negative_mean', ''),
                    stats.get('overall_mean', ''),
                    stats.get('overall_std', ''),
                    stats.get('min', ''),
                    stats.get('max', '')
                ])
            report_data.append([])
            
            # Correlations with PCOS
            report_data.append(['Correlations with PCOS'])
            report_data.append(['Feature', 'Correlation'])
            correlations = uploaded_dataset_analysis.get('correlations', {}).get('with_pcos', {})
            for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                report_data.append([feature, corr])
            
            # Convert to DataFrame and save
            df_report = pd.DataFrame(report_data)
            
            # Create CSV in memory
            buffer = io.StringIO()
            df_report.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            
            # Convert to bytes
            byte_buffer = io.BytesIO(buffer.getvalue().encode())
            byte_buffer.seek(0)
            
            return send_file(
                byte_buffer,
                as_attachment=True,
                download_name=f'pcos_analysis_{timestamp}.csv',
                mimetype='text/csv'
            )
        
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid format. Use "json" or "csv".'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating download: {str(e)}'
        }), 500

@app.route('/api/model-analysis', methods=['POST'])
def get_model_analysis():
    """Get detailed analysis for a specific model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({
                'success': False,
                'message': 'Model name is required'
            }), 400
        
        # Get model details from detailed_analysis
        model_details = detailed_analysis.get(model_name, {})
        
        # Get confusion matrix if available
        confusion_matrix_data = None
        if 'confusion_matrix' in model_details:
            cm = model_details['confusion_matrix']
            if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                confusion_matrix_data = {
                    'tn': int(cm[0][0]),  # True Negatives
                    'fp': int(cm[0][1]),  # False Positives
                    'fn': int(cm[1][0]),  # False Negatives
                    'tp': int(cm[1][1])   # True Positives
                }
        
        # Get feature importances if available
        feature_importances = model_details.get('feature_importances', {})
        
        # Get training info
        training_info = model_details.get('training_info', {})
        
        response_data = {
            'success': True,
            'model_name': model_name,
            'confusion_matrix': confusion_matrix_data,
            'feature_importances': feature_importances,
            'training_info': training_info,
            'strengths': model_details.get('strengths', []),
            'weaknesses': model_details.get('weaknesses', [])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error fetching model analysis: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Loading models...")
    if load_all_models():
        print("Starting Flask server...")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("Failed to load models. Please run train_models.py first.")

# Load models on startup for production servers (gunicorn)
load_all_models()
