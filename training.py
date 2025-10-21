
def plot_confusion_matrix(y_true, y_pred, class_names, fold=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    title = f'Confusion Matrix - Fold {fold}' if fold else 'Confusion Matrix - Final Model'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold_{fold}.png' if fold else 'confusion_matrix_final.png', dpi=300)
    plt.show()

def plot_roc_curves(y_true, y_pred_proba, num_classes, fold=None):
    """Plot ROC curves for multi-class classification"""
    plt.figure(figsize=(12, 8))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, 
                label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = f'ROC Curves - Fold {fold}' if fold else 'ROC Curves - Final Model'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curves_fold_{fold}.png' if fold else 'roc_curves_final.png', dpi=300)
    plt.show()
    
    return roc_auc

# =====================================================
# Main Training Pipeline
# =====================================================
print("="*70)
print("🚀 MULTI-TASK MODEL: CNN + RESIDUAL + KAN")
print("="*70)

# Preprocess data
print("\n📊 Preprocessing data...")
X_train_processed, X_test_processed = advanced_preprocess(X_train, X_test, y_train)

# Prepare labels
y_class = y_train["NACCALZD"].values
y_reg = y_train["CDRSUM"].values.astype(float)

# Encode classification labels
encoder = LabelEncoder()
y_class_enc = encoder.fit_transform(y_class)
num_classes = len(np.unique(y_class_enc))
y_class_cat = to_categorical(y_class_enc, num_classes=num_classes)

print(f"✓ Dataset shape: {X_train_processed.shape}")
print(f"✓ Number of classes: {num_classes}")
print(f"✓ Class distribution: {np.bincount(y_class_enc)}")

# =====================================================
# Stratified K-Fold Cross-Validation
# =====================================================
print(f"\n{'='*70}")
print("🔄 STRATIFIED 5-FOLD CROSS-VALIDATION")
print(f"{'='*70}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_metrics = {
    'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
    'auc_macro': [], 'auc_weighted': [],
    'r2': [], 'mse': [], 'mae': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_class_enc), 1):
    print(f"\n{'='*30} FOLD {fold}/{n_splits} {'='*30}")
    
    # Split data
    X_fold_train, X_fold_val = X_train_processed[train_idx], X_train_processed[val_idx]
    y_class_train, y_class_val = y_class_cat[train_idx], y_class_cat[val_idx]
    y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]
    y_class_enc_val = y_class_enc[val_idx]
    
    # Create model
    model = create_multitask_model(X_train_processed.shape[1], num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_classification_accuracy', patience=20, 
                     restore_best_weights=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                         min_lr=1e-7, verbose=1)
    ]
    
    # Train
    history = model.fit(
        X_fold_train,
        {'classification': y_class_train, 'regression': y_reg_train},
        validation_data=(X_fold_val, {'classification': y_class_val, 'regression': y_reg_val}),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Predictions
    predictions = model.predict(X_fold_val, verbose=0)
    y_class_pred_proba = predictions[0]
    y_class_pred = np.argmax(y_class_pred_proba, axis=1)
    y_reg_pred = predictions[1].flatten()
    
    # Classification metrics
    accuracy = accuracy_score(y_class_enc_val, y_class_pred)
    precision = precision_score(y_class_enc_val, y_class_pred, average='weighted', zero_division=0)
    recall = recall_score(y_class_enc_val, y_class_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_class_enc_val, y_class_pred, average='weighted', zero_division=0)
    
    # AUC scores
    try:
        auc_macro = roc_auc_score(y_class_val, y_class_pred_proba, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_class_val, y_class_pred_proba, average='weighted', multi_class='ovr')
    except:
        auc_macro = 0.0
        auc_weighted = 0.0
    
    # Regression metrics
    r2 = r2_score(y_reg_val, y_reg_pred)
    mse = mean_squared_error(y_reg_val, y_reg_pred)
    mae = np.mean(np.abs(y_reg_val - y_reg_pred))
    
    # Store metrics
    fold_metrics['accuracy'].append(accuracy)
    fold_metrics['precision'].append(precision)
    fold_metrics['recall'].append(recall)
    fold_metrics['f1'].append(f1)
    fold_metrics['auc_macro'].append(auc_macro)
    fold_metrics['auc_weighted'].append(auc_weighted)
    fold_metrics['r2'].append(r2)
    fold_metrics['mse'].append(mse)
    fold_metrics['mae'].append(mae)
    
    # Print results
    print(f"\n📊 FOLD {fold} RESULTS:")
    print(f"  Classification Metrics:")
    print(f"    • Accuracy:        {accuracy:.6f}")
    print(f"    • Precision:       {precision:.6f}")
    print(f"    • Recall:          {recall:.6f}")
    print(f"    • F1-Score:        {f1:.6f}")
    print(f"    • AUC (Macro):     {auc_macro:.6f}")
    print(f"    • AUC (Weighted):  {auc_weighted:.6f}")
    print(f"  Regression Metrics:")
    print(f"    • R² Score:        {r2:.6f}")
    print(f"    • MSE:             {mse:.6f}")
    print(f"    • MAE:             {mae:.6f}")
    
    # Classification report
    print(f"\n  Classification Report:")
    class_names = [f'Class_{i}' for i in range(num_classes)]
    print(classification_report(y_class_enc_val, y_class_pred, 
                              target_names=class_names, digits=4))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_class_enc_val, y_class_pred, class_names, fold=fold)
    
    # Plot ROC curves
    roc_auc_dict = plot_roc_curves(y_class_val, y_class_pred_proba, num_classes, fold=fold)

# =====================================================
# Cross-Validation Summary
# =====================================================
print(f"\n{'='*70}")
print("📈 CROSS-VALIDATION SUMMARY (95% Confidence Intervals)")
print(f"{'='*70}\n")

for metric_name, scores in fold_metrics.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    n = len(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(0.95, n-1, loc=mean_score, scale=sem)
    
    print(f"{metric_name.upper()}:")
    print(f"  Mean: {mean_score:.6f}")
    print(f"  Std:  {std_score:.6f}")
    print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
    print()

# =====================================================
# Train Final Model on Full Training Data
# =====================================================
print(f"\n{'='*70}")
print("🏆 TRAINING FINAL MODEL ON FULL TRAINING DATA")
print(f"{'='*70}\n")

final_model = create_multitask_model(X_train_processed.shape[1], num_classes)

final_callbacks = [
    EarlyStopping(monitor='val_classification_accuracy', patience=25, 
                 restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, 
                     min_lr=1e-7, verbose=1)
]

final_history = final_model.fit(
    X_train_processed,
    {'classification': y_class_cat, 'regression': y_reg},
    validation_split=0.15,
    epochs=50,
    batch_size=16,
    callbacks=final_callbacks,
    verbose=1
)

# Evaluate on training data
train_predictions = final_model.predict(X_train_processed, verbose=0)
y_train_class_pred_proba = train_predictions[0]
y_train_class_pred = np.argmax(y_train_class_pred_proba, axis=1)
y_train_reg_pred = train_predictions[1].flatten()

train_accuracy = accuracy_score(y_class_enc, y_train_class_pred)
train_f1 = f1_score(y_class_enc, y_train_class_pred, average='weighted', zero_division=0)
train_r2 = r2_score(y_reg, y_train_reg_pred)

print(f"\n🎯 FINAL MODEL - TRAINING SET PERFORMANCE:")
print(f"  • Accuracy:  {train_accuracy:.6f}")
print(f"  • F1-Score:  {train_f1:.6f}")
print(f"  • R² Score:  {train_r2:.6f}")

# Final classification report
print(f"\n  Final Classification Report (Training):")
class_names = [f'Class_{i}' for i in range(num_classes)]
print(classification_report(y_class_enc, y_train_class_pred, 
                          target_names=class_names, digits=4))

# Final visualizations
plot_confusion_matrix(y_class_enc, y_train_class_pred, class_names, fold=None)
roc_auc_final = plot_roc_curves(y_class_cat, y_train_class_pred_proba, num_classes, fold=None)

# =====================================================
# Save Model and Results
# =====================================================
final_model.save('multitask_kan_model.keras')
print(f"\n💾 Model saved as 'multitask_kan_model.keras'")

# Save metrics
results_df = pd.DataFrame(fold_metrics)
results_df['fold'] = range(1, n_splits + 1)
results_df.to_csv('cross_validation_results.csv', index=False)
print(f"💾 Cross-validation results saved as 'cross_validation_results.csv'")

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\n📊 Model Architecture:")
print(f"  • 1D CNN with multi-scale kernels (3, 5, 7)")
print(f"  • 3 Residual Blocks (128, 192, 256 filters)")
print(f"  • 2 KAN Layers (192, 128 units)")
print(f"  • Dual Task Heads (Classification + Regression)")
print(f"  • Total Parameters: {final_model.count_params():,}")
print(f"\n🎯 Best Cross-Validation Accuracy: {max(fold_metrics['accuracy']):.6f}")
print(f"🎯 Mean Cross-Validation Accuracy: {np.mean(fold_metrics['accuracy']):.6f} ± {np.std(fold_metrics['accuracy']):.6f}")
print(f"{'='*70}")
