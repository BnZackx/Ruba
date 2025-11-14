# src/train_model.py
"""
Vitamin C Degradation Predictive Model Trainer
BnZackx LIMITED AI Research Division
Author: Umar Faruk Zakariyya
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class VitaminCPredictor:
    """
    AI Model for predicting Vitamin C degradation during thermal processing
    """
    
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.performance_metrics = {}
        
    def load_data(self, file_path):
        """
        Load and preprocess Vitamin C degradation data
        """
        print("📊 Loading dataset...")
        self.df = pd.read_csv(file_path)
        
        # Data validation
        required_columns = ['crop_type', 'temperature_c', 'time_min', 'vitamin_c_mg_per_100g']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"✅ Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        return self.df
    
    def preprocess_data(self):
        """
        Preprocess data for model training
        """
        print("🔧 Preprocessing data...")
        
        # Encode categorical variables
        self.label_encoder = LabelEncoder()
        self.df['crop_encoded'] = self.label_encoder.fit_transform(self.df['crop_type'])
        
        # Create feature matrix
        features = ['temperature_c', 'time_min', 'crop_encoded']
        if 'moisture_content' in self.df.columns:
            features.append('moisture_content')
        
        X = self.df[features]
        y = self.df['vitamin_c_mg_per_100g']
        
        # Create polynomial features
        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = self.poly.fit_transform(X)
        self.feature_names = self.poly.get_feature_names_out(features)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_poly)
        
        print(f"✅ Data preprocessed: {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def train_polynomial_regression(self, X, y, test_size=0.2, random_state=42):
        """
        Train polynomial regression model
        """
        print("🧠 Training Polynomial Regression Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
        test_metrics = self._calculate_metrics(y_test, y_pred_test, "test")
        
        self.performance_metrics.update(train_metrics)
        self.performance_metrics.update(test_metrics)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        self.performance_metrics['cv_r2_mean'] = cv_scores.mean()
        self.performance_metrics['cv_r2_std'] = cv_scores.std()
        
        print("✅ Polynomial Regression training completed")
        return X_test, y_test, y_pred_test
    
    def train_comparative_models(self, X, y):
        """
        Train alternative models for comparison
        """
        print("🔍 Training comparative models...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        
        comparative_results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            comparative_results[name] = metrics
        
        self.performance_metrics['comparative_models'] = comparative_results
        return comparative_results
    
    def _calculate_metrics(self, y_true, y_pred, dataset_type):
        """
        Calculate performance metrics
        """
        return {
            f'{dataset_type}_r2': r2_score(y_true, y_pred),
            f'{dataset_type}_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{dataset_type}_mae': mean_absolute_error(y_true, y_pred),
            f'{dataset_type_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_scores = np.abs(self.model.coef_)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.performance_metrics['feature_importance'] = feature_importance
        return feature_importance
    
    def predict(self, temperature, time, crop_type, moisture=None):
        """
        Predict Vitamin C content for new samples
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode crop type
        crop_encoded = self.label_encoder.transform([crop_type])[0]
        
        # Create feature array
        features = [temperature, time, crop_encoded]
        if moisture is not None:
            features.append(moisture)
        
        X_new = np.array([features])
        
        # Transform features
        X_poly = self.poly.transform(X_new)
        X_scaled = self.scaler.transform(X_poly)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction
    
    def save_model(self, filepath):
        """
        Save trained model and preprocessing objects
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'poly': self.poly,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model and preprocessing objects
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.poly = model_data['poly']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        
        print(f"📂 Model loaded from {filepath}")
    
    def generate_report(self):
        """
        Generate training report
        """
        print("\n" + "="*60)
        print("🤖 VITAMIN C AI PREDICTOR - TRAINING REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total samples: {len(self.df)}")
        print(f"   Crops: {list(self.label_encoder.classes_)}")
        print(f"   Features: {len(self.feature_names)}")
        
        print(f"\n🎯 MODEL PERFORMANCE (Polynomial Regression):")
        for key, value in self.performance_metrics.items():
            if not isinstance(value, dict) and not isinstance(value, pd.DataFrame):
                if 'r2' in key:
                    print(f"   {key}: {value:.4f}")
                elif 'mape' in key:
                    print(f"   {key}: {value:.2f}%")
                else:
                    print(f"   {key}: {value:.2f}")
        
        print(f"\n🔍 COMPARATIVE MODEL PERFORMANCE:")
        if 'comparative_models' in self.performance_metrics:
            for model_name, metrics in self.performance_metrics['comparative_models'].items():
                print(f"   {model_name}:")
                print(f"     R²: {metrics['r2']:.4f}")
                print(f"     RMSE: {metrics['rmse']:.2f}")
                print(f"     MAE: {metrics['mae']:.2f}")
        
        print(f"\n📈 FEATURE IMPORTANCE (Top 10):")
        feature_importance = self.performance_metrics.get('feature_importance')
        if feature_importance is not None:
            print(feature_importance.head(10).to_string(index=False))
        
        print(f"\n© Umar Faruk Zakariyya | BnZackx LIMITED 2024")

def main():
    """
    Main training function
    """
    print("🚀 VITAMIN C AI PREDICTOR - MODEL TRAINING")
    print("="*50)
    
    # Initialize predictor
    predictor = VitaminCPredictor()
    
    try:
        # Load data
        df = predictor.load_data('data/vitamin_c_data.csv')
        
        # Preprocess data
        X, y = predictor.preprocess_data()
        
        # Train polynomial regression
        X_test, y_test, y_pred = predictor.train_polynomial_regression(X, y)
        
        # Train comparative models
        comparative_results = predictor.train_comparative_models(X, y)
        
        # Feature importance analysis
        feature_importance = predictor.feature_importance_analysis()
        
        # Save model
        predictor.save_model('models/vitamin_c_predictor.pkl')
        
        # Generate report
        predictor.generate_report()
        
        # Example predictions
        print(f"\n🔮 SAMPLE PREDICTIONS:")
        samples = [
            (80, 30, 'Orange'),
            (100, 60, 'Baobab'),
            (70, 40, 'Fluted Pumpkin'),
            (90, 20, 'Spinach')
        ]
        
        for temp, time, crop in samples:
            prediction = predictor.predict(temp, time, crop)
            print(f"   {crop} at {temp}°C for {time}min: {prediction:.1f} mg/100g")
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
