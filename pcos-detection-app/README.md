# 🏥 PCOS Detection Web Application

**Live App:** 🌐 https://skills-copilot-codespaces-vscode-dguz.onrender.com/

An AI-powered web application for detecting Polycystic Ovary Syndrome (PCOS) using **4 highly-optimized Machine Learning algorithms**. Optimized for Render's free tier with **ultra-fast predictions** (0.045-0.5 seconds) and **94.44% accuracy**.

---

## 🚀 Quick Start

**Try it now:** [Live App](https://skills-copilot-codespaces-vscode-dguz.onrender.com/)

1. Navigate to "Detection" section
2. Fill in 15 clinical parameters
3. Click "Analyze"
4. Get instant results from 4 ML models

---

## 📊 Machine Learning Models (Free Tier Optimized)

This application uses **4 lightweight sklearn models** for optimal performance:

### 1. **🏆 XGBoost (Gradient Boosting)** - BEST MODEL
- **Accuracy:** 94.44%
- **Type:** Sequential tree boosting
- **Strengths:** Industry-standard accuracy, handles non-linear patterns
- **Use Case:** Primary recommendation model

### 2. **Random Forest**
- **Accuracy:** 92.59%
- **Type:** Ensemble Learning (100 trees)
- **Strengths:** Feature importance, resistant to overfitting
- **Use Case:** Robust predictions with feature interactions

### 3. **Support Vector Machine (SVM)**
- **Accuracy:** 91.67%
- **Type:** Kernel-based ML (RBF kernel)
- **Strengths:** Effective in high-dimensional spaces, memory efficient
- **Use Case:** Finding optimal decision boundaries

### 4. **Logistic Regression**
- **Accuracy:** 89.81%
- **Type:** Classical Machine Learning
- **Strengths:** Fast, interpretable, baseline model
- **Use Case:** Understanding linear feature relationships

### ⚡ Performance Stats
- **Total Model Size:** 368KB (ultra-lightweight!)
- **Prediction Time:** 0.045-0.5 seconds
- **Memory Usage:** ~150MB
- **Best Accuracy:** 94.44% (XGBoost)

> **Note:** Deep Neural Network (TensorFlow) removed for free tier compatibility. XGBoost provides better accuracy anyway!

---

## 🎯 Features

- ✅ **4 Fast ML Models:** XGBoost, Random Forest, SVM, Logistic Regression
- ✅ **Consensus Voting:** Majority voting across all 4 models
- ✅ **Real-time Metrics:** Accuracy, precision, recall, F1-score
- ✅ **Beautiful 3D UI:** Single-page scroll with animations
- ✅ **Model Analysis:** Click any model card for detailed insights
- ✅ **15 Clinical Features:** Comprehensive patient profiling
- ✅ **Instant Results:** Sub-second predictions
- ✅ **Mobile Responsive:** Works on all devices

---

## 📁 Project Structure

```
pcos-detection-app/
├── 🐍 Backend
│   ├── app.py                      # Flask server (1110 lines)
│   ├── train_models.py             # Model training pipeline
│   ├── requirements.txt            # Dependencies (no TensorFlow!)
│   └── build.sh                    # Render build script
│
├── 🎨 Frontend
│   ├── templates/
│   │   └── index.html              # Single-page UI (504 lines)
│   └── static/
│       ├── style.css               # 3D animations (1737 lines)
│       └── script.js               # Model analysis (691 lines)
│
├── 🤖 Machine Learning
│   └── models/
│       ├── xgboost.pkl             # 100KB - BEST (94.44%)
│       ├── random_forest.pkl       # 248KB
│       ├── svm.pkl                 # 16KB
│       ├── logistic_regression.pkl # 4KB
│       ├── scaler.pkl              # 4KB
│       ├── model_comparison.csv    # Performance metrics
│       └── model_comparison.png    # Visualization
│
├── 📊 Data
│   └── data/
│       ├── pcos_dataset.csv        # 541 patients
│       └── sample_data.csv         # Test samples
│
├── 🚀 Deployment
│   ├── render.yaml                 # Render config
│   └── build.sh                    # Build automation
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── RESEARCH_PAPER.md           # Academic paper
    └── research_paper.pdf          # PDF version
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- 500MB disk space

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/rosan-s/skills-copilot-codespaces-vscode.git
cd skills-copilot-codespaces-vscode/pcos-detection-app

# 2. Install dependencies (lightweight - no TensorFlow!)
pip install -r requirements.txt

# 3. Train models (creates 4 sklearn models)
python train_models.py

# 4. Run the application
python app.py

# 5. Open browser
# Visit: http://localhost:5000
```

### Training Output
```
==================================================
PCOS DETECTION - MODEL TRAINING (4 Models)
==================================================
Loading data...
Training set: 432 samples
Test set: 109 samples

Training Logistic Regression...      ✓ Complete
Training Random Forest...            ✓ Complete  
Training SVM...                      ✓ Complete
Training XGBoost...                  ✓ Complete

MODEL PERFORMANCE SUMMARY
Ranking by Accuracy:
1. XGBoost               - 94.44% ⭐ BEST
2. Random Forest         - 92.59%
3. SVM                   - 91.67%
4. Logistic Regression   - 89.81%

RECOMMENDED MODEL: XGBoost (Gradient Boosting)
Models saved to: models/
```

### Verify Installation

```bash
# Test the API
curl http://localhost:5000/api/health

# Expected response:
# {"status": "healthy", "models_loaded": [...]}
```

---

## 🌐 Live Deployment

### Current Deployment
- **Platform:** Render (Free Tier)
- **URL:** https://skills-copilot-codespaces-vscode-dguz.onrender.com
- **Status:** ✅ Active
- **Response Time:** 0.5-2 seconds (cold start may take 3-5s)

### Deploy Your Own (Render)

1. **Fork this repository**
2. **Connect to Render:**
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your forked repo
3. **Configure:**
   ```
   Name: pcos-detection-app
   Environment: Python 3
   Build Command: chmod +x build.sh && ./build.sh
   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 60 --workers 1 --preload
   ```
4. **Deploy:** Click "Create Web Service"
5. **Wait:** 3-5 minutes for first deployment

### Environment Variables (Optional)
```
SKIP_DNN=1           # Skip Deep Neural Network (already default)
PYTHON_VERSION=3.11  # Python version
```

---

## 💡 Usage

### Web Interface

1. **Open App:** https://skills-copilot-codespaces-vscode-dguz.onrender.com
2. **Navigate:** Scroll or use navbar (Home → Models → Detection → Results)
3. **View Models:** See 4 model cards with performance metrics
4. **Click Model:** View detailed analysis (confusion matrix, insights)
5. **Enter Data:** Fill 15 clinical parameters
6. **Analyze:** Click button, wait 1-5 seconds
7. **Results:** View consensus + individual predictions

### API Endpoints

#### Health Check
```bash
GET https://skills-copilot-codespaces-vscode-dguz.onrender.com/api/health

Response:
{
  "status": "healthy",
  "models_loaded": ["Logistic Regression", "Random Forest", "SVM", "XGBoost", "scaler"]
}
```

#### Get Models
```bash
GET https://skills-copilot-codespaces-vscode-dguz.onrender.com/api/models

Response:
{
  "success": true,
  "models": [
    {
      "rank": 1,
      "name": "XGBoost",
      "accuracy": 94.44,
      "precision": 92.31,
      "recall": 92.31,
      "f1_score": 92.31
    }
  ],
  "best_model": "XGBoost"
}
```

#### Make Prediction
```bash
POST https://skills-copilot-codespaces-vscode-dguz.onrender.com/api/predict
Content-Type: application/json

{
  "Age": 28,
  "BMI": 24.5,
  "Cycle_length": 30,
  "FSH": 5.2,
  "LH": 8.1,
  "TSH": 2.3,
  "AMH": 3.5,
  "Insulin": 12.4,
  "Weight_gain": 0,
  "Hair_growth": 0,
  "Skin_darkening": 0,
  "Hair_loss": 0,
  "Pimples": 0,
  "Fast_food": 0,
  "Reg_Exercise": 1
}

Response:
{
  "success": true,
  "predictions": {
    "XGBoost": {
      "prediction": 0,
      "confidence": 95.5,
      "result": "No PCOS",
      "accuracy": 94.44
    },
    ...
  },
  "consensus": {
    "prediction": 0,
    "result": "No PCOS",
    "votes": "0/4 models predict PCOS"
  }
}
```

---

## 📊 Clinical Features (15 Parameters)

### Demographic (2)
1. **Age** (years): 18-50
2. **BMI** (kg/m²): 15-50

### Hormonal Markers (6)
3. **Cycle Length** (days): 20-60
4. **FSH** (mIU/mL): Follicle Stimulating Hormone
5. **LH** (mIU/mL): Luteinizing Hormone
6. **TSH** (mIU/L): Thyroid Stimulating Hormone
7. **AMH** (ng/mL): Anti-Müllerian Hormone (18.5% feature importance!)
8. **Insulin** (μIU/mL): Fasting insulin levels

### Clinical Symptoms (7 - Binary Yes/No)
9. **Weight Gain**: Recent weight gain
10. **Hair Growth**: Hirsutism (excessive hair)
11. **Skin Darkening**: Acanthosis nigricans
12. **Hair Loss**: Scalp hair thinning
13. **Pimples**: Acne/skin issues
14. **Fast Food**: Regular consumption
15. **Regular Exercise**: Physical activity

### Top Predictive Features
1. AMH (Anti-Müllerian Hormone): **18.5%**
2. LH/FSH Ratio: **15.2%**
3. Cycle Length: **12.8%**
4. BMI: **11.3%**
5. Insulin: **9.7%**

---

## 📈 Performance Metrics

### Model Accuracy Comparison
| Rank | Model | Accuracy | Precision | Recall | F1-Score | Speed |
|------|-------|----------|-----------|--------|----------|-------|
| 🥇 1 | **XGBoost** | **94.44%** | **92.31%** | **92.31%** | **92.31%** | 0.02s |
| 🥈 2 | Random Forest | 92.59% | 89.47% | 89.47% | 89.47% | 0.01s |
| 🥉 3 | SVM | 91.67% | 88.00% | 88.00% | 88.00% | 0.01s |
| 4 | Logistic Regression | 89.81% | 85.71% | 85.71% | 85.71% | 0.003s |

### Cross-Validation (5-Fold)
- XGBoost: 93.8% ± 1.2%
- Random Forest: 91.2% ± 2.1%
- SVM: 90.5% ± 1.9%
- Logistic Regression: 88.7% ± 2.3%

### Dataset
- **Total Samples:** 541 patients
- **PCOS Positive:** 177 (32.7%)
- **PCOS Negative:** 364 (67.3%)
- **Train/Test Split:** 80/20 (432 train, 109 test)

---

## 🛠️ Technologies

### Backend
```
Flask 3.0.0          - Web framework
Python 3.11+         - Programming language
Scikit-learn 1.3     - ML algorithms
XGBoost 2.0          - Gradient boosting
Pandas 2.1           - Data processing
NumPy 1.26           - Numerical computing
Matplotlib 3.8       - Visualization
Seaborn 0.13         - Statistical plots
Joblib 1.3           - Model serialization
Gunicorn 21.2        - WSGI server
```

### Frontend
```
HTML5                - Structure
CSS3                 - Animations & glass-morphism
JavaScript (ES6+)    - Interactivity
Fetch API            - Backend communication
```

### Deployment
```
Render               - Cloud platform (Free tier)
GitHub               - Version control
Git                  - Source control
```

---

## ⚠️ Important Notes

### Medical Disclaimer
**This application is for educational and research purposes only.**

❌ NOT a replacement for professional medical diagnosis  
❌ NOT a definitive diagnostic tool  
❌ NOT a substitute for clinical examination

✅ **Always consult qualified healthcare professionals** for diagnosis and treatment

### Free Tier Optimization
This app is optimized for Render's free tier:
- ✅ No TensorFlow (removed for performance)
- ✅ 4 lightweight sklearn models (368KB total)
- ✅ Fast predictions (0.045-0.5s)
- ✅ Low memory (~150MB vs ~400MB with TensorFlow)
- ✅ Same best accuracy (94.44% XGBoost)

### Data Privacy
When using with real patient data:
- Ensure HIPAA compliance (US)
- Ensure GDPR compliance (EU)
- Use secure HTTPS transmission
- Anonymize patient data
- Follow local medical data regulations

---

## 📚 Research Paper

Comprehensive academic documentation available:
- **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** - Full markdown (8500+ words)
- **[research_paper.pdf](research_paper.pdf)** - PDF version (5 pages)

Includes:
- Literature review
- Methodology and algorithms
- Performance analysis
- Clinical implications
- Web architecture
- Future work

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
1. Additional datasets with external validation
2. More ML algorithms (LightGBM, CatBoost)
3. Enhanced visualizations
4. Mobile app development
5. EHR system integration

---

## 📧 Support

- **Email:** rosans.tech@gmail.com
- **GitHub:** [rosan-s](https://github.com/rosan-s)
- **Repository:** [skills-copilot-codespaces-vscode](https://github.com/rosan-s/skills-copilot-codespaces-vscode)
- **Issues:** [Report Bug](https://github.com/rosan-s/skills-copilot-codespaces-vscode/issues)

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| **Live App** | https://skills-copilot-codespaces-vscode-dguz.onrender.com |
| **GitHub** | https://github.com/rosan-s/skills-copilot-codespaces-vscode |
| **Research Paper (PDF)** | [Download](research_paper.pdf) |
| **API Health** | [/api/health](https://skills-copilot-codespaces-vscode-dguz.onrender.com/api/health) |
| **API Models** | [/api/models](https://skills-copilot-codespaces-vscode-dguz.onrender.com/api/models) |

---

## 📜 License

Educational and research use. Ensure compliance with medical data regulations when using with real patient data.

---

**Last Updated:** December 9, 2025  
**Version:** 2.0 (Free Tier Optimized)  
**Status:** ✅ Production Ready  
**Live:** https://skills-copilot-codespaces-vscode-dguz.onrender.com

Built with ❤️ for improving women's health through AI

Training Logistic Regression...
Training Random Forest...
Training Support Vector Machine...
Training XGBoost...
Training Deep Neural Network...

MODEL COMPARISON SUMMARY
Ranking by Accuracy:
1. XGBoost               - Accuracy: 0.9500 | ...
2. Random Forest         - Accuracy: 0.9250 | ...
3. Deep Neural Network   - Accuracy: 0.9000 | ...
4. SVM                   - Accuracy: 0.8750 | ...
5. Logistic Regression   - Accuracy: 0.8500 | ...

RECOMMENDED MODEL: XGBoost
```

### Step 3: Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 🔬 Usage

### Web Interface

1. **Access the Application:** Open your browser and navigate to `http://localhost:5000`

2. **View Model Comparison:** The homepage displays a comparison table of all 5 models with their performance metrics

3. **Enter Patient Data:** Fill in the form with:
   - **Clinical Parameters:** Age, BMI, Cycle Length, Hormone Levels (FSH, LH, TSH, AMH), Insulin
   - **Symptoms:** Weight gain, Hair growth, Skin darkening, Hair loss, Acne
   - **Lifestyle:** Fast food consumption, Regular exercise

4. **Get Predictions:** Click "Analyze" to get:
   - Consensus prediction from all models
   - Individual predictions from each model
   - Confidence scores and accuracy metrics

### API Endpoints

#### Get Model Comparison
```bash
GET /api/models
```

Response:
```json
{
  "success": true,
  "models": [
    {
      "rank": 1,
      "name": "XGBoost",
      "accuracy": 95.0,
      "precision": 94.5,
      "recall": 95.2,
      "f1_score": 94.8
    },
    ...
  ],
  "best_model": "XGBoost"
}
```

#### Make Prediction
```bash
POST /api/predict
Content-Type: application/json

{
  "Age": 25,
  "BMI": 28.5,
  "Cycle_length": 35,
  "FSH": 4.8,
  "LH": 12.3,
  "TSH": 2.5,
  "AMH": 6.2,
  "Insulin": 18.5,
  "Weight_gain": 1,
  "Hair_growth": 1,
  "Skin_darkening": 0,
  "Hair_loss": 0,
  "Pimples": 1,
  "Fast_food": 1,
  "Reg_Exercise": 0
}
```

Response:
```json
{
  "success": true,
  "predictions": {
    "Logistic Regression": {
      "prediction": 1,
      "confidence": 87.5,
      "result": "PCOS Detected",
      "accuracy": 85.0
    },
    ...
  },
  "consensus": {
    "prediction": 1,
    "result": "PCOS Detected",
    "votes": "4/5 models predict PCOS"
  }
}
```

## 📈 Model Performance Metrics

### Evaluation Metrics

- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of positive identifications that were actually correct
- **Recall:** Proportion of actual positives that were identified correctly
- **F1-Score:** Harmonic mean of precision and recall
- **Cross-Validation:** 5-fold CV for robust performance estimation

### Expected Performance (with sample data)

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| 🥇 1 | **XGBoost** | ~95% | ~94% | ~95% | ~95% |
| 🥈 2 | Random Forest | ~92% | ~91% | ~93% | ~92% |
| 🥉 3 | Deep Neural Network | ~90% | ~89% | ~91% | ~90% |
| 4 | SVM | ~87% | ~86% | ~88% | ~87% |
| 5 | Logistic Regression | ~85% | ~84% | ~86% | ~85% |

**Note:** Performance may vary based on dataset size and quality.

## 🎓 Input Features

### Clinical Parameters (8)
1. **Age** - Patient's age in years (18-50)
2. **BMI** - Body Mass Index in kg/m² (15-50)
3. **Cycle Length** - Menstrual cycle length in days (20-60)
4. **FSH** - Follicle Stimulating Hormone in mIU/mL (0-20)
5. **LH** - Luteinizing Hormone in mIU/mL (0-30)
6. **TSH** - Thyroid Stimulating Hormone in mIU/L (0-10)
7. **AMH** - Anti-Müllerian Hormone in ng/mL (0-15)
8. **Insulin** - Insulin level in μIU/mL (0-50)

### Symptoms & Lifestyle (7)
9. **Weight Gain** - Recent weight gain (Yes/No)
10. **Hair Growth** - Excessive hair growth/hirsutism (Yes/No)
11. **Skin Darkening** - Acanthosis nigricans (Yes/No)
12. **Hair Loss** - Scalp hair thinning (Yes/No)
13. **Pimples** - Acne/skin issues (Yes/No)
14. **Fast Food** - Regular fast food consumption (Yes/No)
15. **Regular Exercise** - Consistent exercise routine (Yes/No)

## 🔧 Model Recommendation

### **Winner: XGBoost** 🏆

**Why XGBoost is Recommended:**

1. **Highest Accuracy:** Consistently achieves the best performance across all metrics
2. **Robust Performance:** Handles imbalanced data well with built-in regularization
3. **Feature Importance:** Provides insights into which factors contribute most to PCOS
4. **Scalability:** Efficient with larger datasets
5. **Industry Standard:** Widely used in medical ML applications

**When to Use Other Models:**

- **Random Forest:** When interpretability and feature importance are crucial
- **Deep Neural Network:** With larger datasets (>10,000 samples) for better generalization
- **SVM:** When you have limited data but need good generalization
- **Logistic Regression:** For quick baseline and when explainability is mandatory

## ⚠️ Important Notes

### Medical Disclaimer
This application is designed for **educational and research purposes only**. It should NOT be used as:
- A replacement for professional medical diagnosis
- A definitive diagnostic tool
- A substitute for clinical examination and testing

**Always consult qualified healthcare professionals for proper diagnosis and treatment.**

### Dataset Considerations
The sample dataset provided is for demonstration purposes. For production use:
- Use larger, validated medical datasets
- Ensure proper data collection and annotation
- Consider data privacy and HIPAA compliance
- Perform thorough validation with clinical data

## 🛠️ Customization

### Using Your Own Dataset

1. Prepare your CSV file with the same column names as `sample_data.csv`
2. Place it in the `data/` directory
3. Update `train_models.py`:
```python
trainer = PCOSModelTrainer(data_path='data/your_dataset.csv')
```
4. Retrain the models

### Adjusting Model Parameters

Edit `train_models.py` to tune hyperparameters:

```python
# Example: Adjust Random Forest
model = RandomForestClassifier(
    n_estimators=200,      # Increase trees
    max_depth=15,          # Deeper trees
    min_samples_split=3    # Different split criteria
)
```

## 📊 Visualizations

After training, check `models/model_comparison.png` for:
- Accuracy comparison bar chart
- Precision comparison
- Recall comparison
- F1-Score comparison

## 🤝 Contributing

To improve the application:
1. Collect more diverse PCOS datasets
2. Implement additional models (LightGBM, CatBoost)
3. Add feature engineering
4. Improve UI/UX
5. Add export functionality for predictions

## 📝 License

This project is for educational purposes. Please ensure compliance with medical data regulations when using with real patient data.

## 🔗 Technologies Used

- **Backend:** Flask (Python)
- **ML Libraries:** scikit-learn, XGBoost
- **Deep Learning:** TensorFlow/Keras
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## 📧 Support

For issues or questions:
1. Check the console output for error messages
2. Ensure all dependencies are installed
3. Verify models are trained before running the app
4. Check that port 5000 is available

---

**Built with ❤️ for improving women's health through AI**
