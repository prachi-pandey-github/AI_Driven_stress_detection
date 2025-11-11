import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class RealStressDetector:
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.dataset_info = {}
        
    def load_and_explore_dataset(self, file_path):
        """
        Load and explore any mental health dataset
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✅ Successfully loaded dataset with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                # If encodings fail, try without specific encoding
                df = pd.read_csv(file_path)
            
            print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Explore the dataset
            self.explore_dataset(df)
            
            return df
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            print("\n💡 Please download a mental health dataset from Kaggle such as:")
            print("   - Mental Health in Tech Survey: https://www.kaggle.com/datasets/catherinehorsey/mental-health-in-tech-survey")
            print("   - Student Stress Factors: https://www.kaggle.com/datasets/laavanya/student-stress-factors")
            print("   - Save it in your project folder and update the file path")
            return None
    
    def explore_dataset(self, df):
        """Explore the dataset structure"""
        print("\n🔍 DATASET EXPLORATION")
        print("=" * 50)
        
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nMissing Values:")
        missing_data = df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        # Store dataset info
        self.dataset_info['columns'] = list(df.columns)
        self.dataset_info['shape'] = df.shape
        self.dataset_info['dtypes'] = df.dtypes.to_dict()
    
    def auto_preprocess_dataset(self, df):
        """
        Automatically preprocess the dataset based on common mental health survey structures
        """
        df_processed = df.copy()
        
        print("\n🔄 Auto-preprocessing dataset...")
        
        # 1. Identify potential target column
        target_column = self.identify_target_column(df_processed)
        if target_column:
            print(f"🎯 Identified target column: {target_column}")
        else:
            # Create a synthetic target if none found
            print("⚠️ No clear target column found. Creating synthetic target...")
            target_column = 'stress_risk'
            df_processed[target_column] = self.create_synthetic_target(df_processed)
        
        # 2. Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # 3. Remove unnecessary columns
        df_processed = self.remove_unnecessary_columns(df_processed, target_column)
        
        print(f"✅ Preprocessing complete. Final shape: {df_processed.shape}")
        return df_processed, target_column
    
    def identify_target_column(self, df):
        """Identify the most likely target column"""
        target_keywords = [
            'stress', 'depression', 'anxiety', 'mental', 'treatment', 
            'diagnosis', 'condition', 'label', 'target', 'result'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    return col
        
        # Check for binary columns that could be targets
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() == 2:
                unique_vals = df[col].unique()
                if set(unique_vals) in [{'Yes', 'No'}, {'YES', 'NO'}, {'yes', 'no'}, {'1', '0'}, {1, 0}]:
                    return col
        
        return None
    
    def create_synthetic_target(self, df):
        """Create a synthetic target variable based on available features"""
        # Simple heuristic: if person has multiple risk factors, mark as high stress
        risk_score = 0
        
        # Check for common risk indicators
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Assume higher values in numeric columns might indicate stress
                if df[col].max() > df[col].min():  # Avoid constant columns
                    risk_score += (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Create binary target (top 30% as high stress)
        threshold = risk_score.quantile(0.7)
        return (risk_score > threshold).astype(int)
    
    def handle_missing_values(self, df):
        """Handle missing values appropriately"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype == 'object':
                    # For categorical, use mode or 'Unknown'
                    df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
                else:
                    # For numerical, use median
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        return df_clean
    
    def remove_unnecessary_columns(self, df, target_column):
        """Remove columns that are not useful for modeling"""
        df_clean = df.copy()
        
        columns_to_remove = []
        
        for col in df_clean.columns:
            # Remove constant columns
            if df_clean[col].nunique() <= 1:
                columns_to_remove.append(col)
            # Remove high-cardinality categorical columns (unless it's the target)
            elif df_clean[col].dtype == 'object' and df_clean[col].nunique() > 50 and col != target_column:
                columns_to_remove.append(col)
            # Remove ID-like columns
            elif any(keyword in col.lower() for keyword in ['id', 'name', 'email', 'timestamp']):
                columns_to_remove.append(col)
        
        if columns_to_remove:
            print(f"🗑️ Removing columns: {columns_to_remove}")
            df_clean = df_clean.drop(columns=columns_to_remove)
        
        return df_clean
    
    def prepare_features(self, df, target_column):
        """Prepare features for modeling"""
        df_processed = df.copy()
        
        # Separate target
        y = df_processed[target_column]
        X = df_processed.drop(columns=[target_column])
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0, 'yes': 1, 'no': 0}).fillna(0)
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            try:
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            except:
                # If encoding fails, use simple integer encoding
                X[col] = X[col].astype('category').cat.codes
        
        # Handle numerical columns - fill any remaining missing values
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            X[col] = X[col].fillna(X[col].median())
        
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Features prepared: {len(self.feature_names)} features")
        print(f"📋 Feature names: {self.feature_names}")
        
        return X, y
    
    def train_model(self, df, target_column):
        """Train logistic regression model"""
        print("\n🔄 Preparing features for modeling...")
        X, y = self.prepare_features(df, target_column)
        
        # Check if we have enough data
        if len(X) < 50:
            print("❌ Not enough data for training. Need at least 50 samples.")
            return None, None, 0, 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        print("📏 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("🤖 Training Logistic Regression...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Model training completed!")
        print(f"🎯 Accuracy: {accuracy:.3f}")
        print(f"📊 AUC Score: {auc_score:.3f}")
        
        return X_test_scaled, y_test, accuracy, auc_score
    
    def plot_feature_importance(self):
        """Plot feature importance from logistic regression"""
        if not hasattr(self.model, 'coef_'):
            print("❌ Model not trained yet!")
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_importance': np.abs(self.model.coef_[0])
        }).sort_values('abs_importance', ascending=True)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if x > 0 else 'green' for x in importance['coefficient']]
        bars = plt.barh(importance['feature'], importance['coefficient'], color=colors, alpha=0.7)
        
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.title('Feature Impact on Stress Risk\n(Red = Increases Risk, Green = Decreases Risk)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        
        # Add value labels
        for bar, coeff in zip(bars, importance['coefficient']):
            plt.text(bar.get_width() + (0.01 if coeff > 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{coeff:.3f}', 
                    ha='left' if coeff > 0 else 'right', 
                    va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return importance
    
    def create_example_prediction(self):
        """Create an example prediction based on actual features"""
        if not self.feature_names:
            print("❌ No features available. Train model first.")
            return
        
        print(f"\n🎯 Creating example prediction with {len(self.feature_names)} features...")
        
        # Create realistic example based on feature types
        example_features = {}
        
        for feature in self.feature_names:
            if feature in self.label_encoders:
                # For categorical features, use the first category
                categories = self.label_encoders[feature].classes_
                example_features[feature] = categories[0] if len(categories) > 0 else "Unknown"
            else:
                # For numerical features, use median value
                example_features[feature] = 0  # Will be scaled anyway
        
        return example_features
    
    def predict_individual(self, input_features):
        """Predict stress risk for an individual"""
        if not self.feature_names:
            raise ValueError("❌ Model not trained yet!")
        
        # Create input DataFrame with correct feature order
        input_df = pd.DataFrame([input_features])
        
        # Ensure all features are present, fill missing with defaults
        for feature in self.feature_names:
            if feature not in input_df.columns:
                if feature in self.label_encoders:
                    # For categorical, use first category
                    categories = self.label_encoders[feature].classes_
                    input_df[feature] = categories[0] if len(categories) > 0 else "Unknown"
                else:
                    # For numerical, use 0
                    input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[self.feature_names]
        
        # Encode categorical variables
        for col in self.feature_names:
            if col in self.label_encoders:
                input_value = input_df[col].iloc[0]
                if input_value in self.label_encoders[col].classes_:
                    input_df[col] = self.label_encoders[col].transform([input_value])[0]
                else:
                    # Use first category for unseen labels
                    input_df[col] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0][1]
        
        return {
            'stress_risk': 'High' if prediction == 1 else 'Low',
            'probability': probability,
            'confidence': 'High' if probability > 0.7 else 'Medium' if probability > 0.5 else 'Low'
        }

def main():
    """Main function to run the stress detection system"""
    print("🧠 AUTOMATIC STRESS DETECTION SYSTEM")
    print("====================================\n")
    
    # Initialize detector
    detector = RealStressDetector()
    
    # Load dataset - UPDATE THIS PATH TO YOUR DATASET
    file_path = "data/mental_health.csv"  # Change this to your actual file path
    
    print(f"📥 Loading dataset from: {file_path}")
    df = detector.load_and_explore_dataset(file_path)
    
    if df is None:
        print("\n💡 Example file paths to try:")
        print("   - 'survey.csv' (Mental Health in Tech Survey)")
        print("   - 'stress.csv' (Student Stress Factors)")
        print("   - 'data/mental_health.csv' (if in data folder)")
        return
    
    # Auto-preprocess dataset
    df_processed, target_column = detector.auto_preprocess_dataset(df)
    
    # Train model
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    X_test, y_test, accuracy, auc = detector.train_model(df_processed, target_column)
    
    if X_test is None:
        return
    
    # Feature importance
    print("\n" + "="*50)
    print("FEATURE ANALYSIS")
    print("="*50)
    
    importance_df = detector.plot_feature_importance()
    
    if importance_df is not None:
        print("\n🔍 Top factors increasing stress risk:")
        top_positive = importance_df[importance_df['coefficient'] > 0].nlargest(5, 'coefficient')
        print(top_positive[['feature', 'coefficient']].to_string(index=False))
        
        print("\n🛡️ Top factors decreasing stress risk:")
        top_negative = importance_df[importance_df['coefficient'] < 0].nsmallest(5, 'coefficient')
        print(top_negative[['feature', 'coefficient']].to_string(index=False))
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    example_features = detector.create_example_prediction()
    if example_features:
        print(f"📋 Example features: {example_features}")
        
        prediction = detector.predict_individual(example_features)
        print(f"\n🎯 Prediction: {prediction['stress_risk']} Stress Risk")
        print(f"📊 Probability: {prediction['probability']:.1%}")
        print(f"💪 Confidence: {prediction['confidence']}")
    
    print("\n" + "="*50)
    print("SYSTEM READY!")
    print("="*50)
    print("✅ Stress detection system is successfully trained and ready!")
    print(f"✅ Using {len(detector.feature_names)} features")
    print(f"✅ Model accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()