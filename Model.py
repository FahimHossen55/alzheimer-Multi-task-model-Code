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

class KANLayer(Layer):
    """Kolmogorov-Arnold Network Layer"""
    def __init__(self, units, num_basis=5, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.num_basis = num_basis
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Basis function weights
        self.basis_weights = self.add_weight(
            name='basis_weights',
            shape=(input_dim, self.num_basis, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Combination weights
        self.combination_weights = self.add_weight(
            name='combination_weights',
            shape=(self.num_basis, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        super(KANLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Apply basis functions (using Chebyshev-like polynomials)
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]
        
        # Expand inputs for basis computation
        x_expanded = tf.expand_dims(inputs, axis=-1)  # [batch, input_dim, 1]
        
        # Compute polynomial basis
        basis_outputs = []
        for i in range(self.num_basis):
            power = float(i)
            if i == 0:
                basis = tf.ones_like(x_expanded)
            else:
                basis = tf.pow(x_expanded, power)
            basis_outputs.append(basis)
        
        basis_stack = tf.concat(basis_outputs, axis=-1)  # [batch, input_dim, num_basis]
        
        # Apply basis weights: [batch, input_dim, num_basis] x [input_dim, num_basis, units]
        # Result: [batch, input_dim, units]
        weighted_basis = tf.einsum('bik,ikj->bij', basis_stack, self.basis_weights)
        
        # Sum across input dimensions: [batch, input_dim, units] -> [batch, units]
        output = tf.reduce_sum(weighted_basis, axis=1)
        
        # Add bias
        output = output + self.bias
        
        return tf.nn.gelu(output)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def get_config(self):
        config = super(KANLayer, self).get_config()
        config.update({
            'units': self.units,
            'num_basis': self.num_basis
        })
        return config
  
        

# =====================================================
# Residual Block with 1D CNN
# =====================================================
def residual_block(x, filters, kernel_size=3, dropout_rate=0.3):
    """Enhanced Residual Block with 1D CNN"""
    # First conv layer
    res = Conv1D(filters, kernel_size, padding='same', 
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    res = BatchNormalization()(res)
    res = tf.keras.layers.Activation('gelu')(res)
    res = Dropout(dropout_rate)(res)
    
    # Second conv layer
    res = Conv1D(filters, kernel_size, padding='same',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(res)
    res = BatchNormalization()(res)
    
    # Match dimensions for skip connection
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    
    # Add skip connection
    output = Add()([x, res])
    output = tf.keras.layers.Activation('gelu')(output)
    output = Dropout(dropout_rate * 0.5)(output)
    
    return output

# =====================================================
# Advanced Preprocessing
# =====================================================

# =====================================================
# Fast Advanced Preprocessing
# =====================================================

    
   


# =====================================================
# Multi-Task Model with CNN, Residual & KAN
# =====================================================
def create_multitask_model(input_dim, num_classes):
    """
    Multi-task model with:
    - 1D CNN layers
    - Residual blocks
    - KAN layers
    - Dual heads (classification + regression)
    """
    inputs = Input(shape=(input_dim,))
    
    # ==================== CNN Branch ====================
    # Reshape for CNN
    x_cnn = Reshape((input_dim, 1))(inputs)
    
    # Multi-scale CNN with different kernel sizes
    cnn_branches = []
    for kernel_size in [3, 5, 7]:
        branch = Conv1D(64, kernel_size, padding='same', activation='gelu',
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x_cnn)
        branch = BatchNormalization()(branch)
        branch = MaxPooling1D(pool_size=2)(branch)
        cnn_branches.append(branch)
    
    x_cnn = Concatenate()(cnn_branches)
    
    # Residual blocks
    x_cnn = residual_block(x_cnn, 128, kernel_size=3, dropout_rate=0.3)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = residual_block(x_cnn, 192, kernel_size=3, dropout_rate=0.3)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = residual_block(x_cnn, 256, kernel_size=3, dropout_rate=0.25)
    
    # Global pooling
    cnn_features = GlobalAveragePooling1D()(x_cnn)
    
    # ==================== KAN Branch ====================
    x_kan = Dense(256, activation='gelu')(inputs)
    x_kan = BatchNormalization()(x_kan)
    x_kan = Dropout(0.3)(x_kan)
    
    # KAN layers
    x_kan = KANLayer(units=192, num_basis=5)(x_kan)
    x_kan = BatchNormalization()(x_kan)
    x_kan = Dropout(0.3)(x_kan)
    
    x_kan = KANLayer(units=128, num_basis=5)(x_kan)
    x_kan = BatchNormalization()(x_kan)
    x_kan = Dropout(0.25)(x_kan)
    
    # ==================== Feature Fusion ====================
    combined = Concatenate()([cnn_features, x_kan])
    
    # Fusion with residual
    x = Dense(384, activation='gelu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Residual connection in dense layers
    skip = Dense(192, activation='gelu')(combined)
    x = Dense(192, activation='gelu')(x)
    x = Add()([x, skip])
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Additional processing
    x = Dense(128, activation='gelu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # ==================== Classification Head ====================
    class_head = Dense(96, activation='gelu')(x)
    class_head = BatchNormalization()(class_head)
    class_head = Dropout(0.2)(class_head)
    class_head = Dense(64, activation='gelu')(class_head)
    class_head = Dropout(0.15)(class_head)
    class_output = Dense(num_classes, activation='softmax', name='classification')(class_head)
    
    # ==================== Regression Head ====================
    reg_head = Dense(96, activation='gelu')(x)
    reg_head = BatchNormalization()(reg_head)
    reg_head = Dropout(0.2)(reg_head)
    reg_head = Dense(64, activation='gelu')(reg_head)
    reg_head = Dropout(0.15)(reg_head)
    reg_output = Dense(1, activation='linear', name='regression')(reg_head)
    
    # Create model
    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    
    # Compile
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss={
            'classification': 'categorical_crossentropy',
            'regression': 'huber'
        },
        loss_weights={
            'classification': 1.5,
            'regression': 0.8
        },
        metrics={
            'classification': ['accuracy'],
            'regression': ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
                          tf.keras.metrics.R2Score(name='r2_score')]
        }
    )
    
    return model