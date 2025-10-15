## 🤖 Model Information
Algorithm: Random Forest Classifier

Accuracy: ~75-80%

Training: 80-20 train-test split with stratification

## 📊 Dataset
The Pima Indians Diabetes Dataset contains 768 patient records with 8 health metrics and a diabetes outcome.

### Features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic predisposition)
- **Age**: Age in years

### Target:
- **Outcome**: Binary classification (0 = No Diabetes, 1 = Diabetes)

## 🎯 Usage
Navigate to Prediction Page: Use the sidebar to select "🎯 Prediction"

Input Health Parameters: Adjust sliders for all 8 health metrics

Get Instant Results: View diabetes risk probability and recommendations

Explore Data: Use other pages for data analysis and model insights

## ☁️ Deployment
Streamlit Cloud Deployment:
Push code to GitHub repository

Visit share.streamlit.io

Connect GitHub account and select repository

Deploy with main file path: app.py

## ⚠️ Medical Disclaimer
This application is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 🔧 Technologies Used
Frontend: Streamlit

Backend: Python, Scikit-learn

Visualization: Plotly, Matplotlib, Seaborn

Data Processing: Pandas, NumPy

## 🎯 Key Customizations:
Medical-focused terminology and icons (🩺 instead of 🍷)

Diabetes-specific features and dataset description

Health parameter details instead of wine chemical properties

Medical disclaimer prominently displayed

Appropriate target variable (Diabetes Outcome vs Wine Quality)

Healthcare-themed technology stack

