
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            mean_squared_error, r2_score, classification_report,
                            confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.impute import KNNImputer
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                   Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                   Add, Multiply, Concatenate, Reshape, Layer)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def advanced_preprocess(X_train, X_test, y_train):
    """Fast preprocessing with smart imputation"""
    print("📊 Starting preprocessing...")
    
    # Clean column names
    X_train.columns = X_train.columns.str.strip()
    X_test.columns = X_test.columns.str.strip()
    y_train.columns = y_train.columns.str.strip()
    
    # Combine for consistent preprocessing
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    
    # Handle duplicates
    if X_combined.columns.duplicated().any():
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        print(f"   ✓ Removed duplicate columns")
    
    # Replace infinities
    X_combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Identify numeric columns
    numeric_cols = X_combined.select_dtypes(include=[np.number]).columns
    print(f"   ✓ Processing {len(numeric_cols)} numeric columns...")
    
    # Strategy 1: Clip extreme outliers (1st to 99th percentile)
    for col in numeric_cols:
        q1 = X_combined[col].quantile(0.01)
        q99 = X_combined[col].quantile(0.99)
        X_combined[col] = X_combined[col].clip(lower=q1, upper=q99)
    print(f"   ✓ Outlier clipping complete")
    
    # Strategy 2: Fast smart imputation
    print(f"   ✓ Imputing missing values...")
    # Store original missing patterns before imputation
    original_missing_dict = {}
    for col in numeric_cols:
        original_missing_dict[col] = X_combined[col].isnull().copy()
    
    for col in numeric_cols:
        if X_combined[col].isnull().sum() > 0:
            # Use median for columns with <30% missing, mean for others
            missing_pct = X_combined[col].isnull().sum() / len(X_combined)
            if missing_pct < 0.3:
                fill_value = X_combined[col].median()
            else:
                fill_value = X_combined[col].mean()
            
            # Fallback to 0 if still NaN
            if pd.isna(fill_value):
                fill_value = 0
            
            X_combined[col].fillna(fill_value, inplace=True)
    print(f"   ✓ Imputation complete")
    
    # Strategy 3: Add missing value indicators for important features
    missing_indicators = []
    for col in numeric_cols:
        original_missing = original_missing_dict[col]
        missing_count = original_missing.sum()  # This is now a scalar
        if missing_count > len(X_combined) * 0.05:  # >5% missing
            missing_indicators.append(original_missing.astype(int).values)
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_combined[numeric_cols])
    
    # Add missing indicators as extra features
    if missing_indicators:
        missing_array = np.column_stack(missing_indicators)
        X_scaled = np.hstack([X_scaled, missing_array])
        print(f"   ✓ Added {len(missing_indicators)} missing value indicators")
    
    print(f"   ✓ Scaling complete")
    
    # Split back
    train_size = len(X_train)
    X_train_processed = X_scaled[:train_size]
    X_test_processed = X_scaled[train_size:]
    
    print(f"✅ Preprocessing finished!")
    print(f"   Training shape: {X_train_processed.shape}")
    print(f"   Test shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed