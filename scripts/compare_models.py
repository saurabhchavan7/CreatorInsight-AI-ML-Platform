"""
Compare staging (newly trained) model vs production (current) model
Only promotes if new model is significantly better
"""

import mlflow
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
import sys
import os

# Import from config
try:
    from config import MLFLOW_TRACKING_URI, MODEL_NAME
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
except ImportError:
    # Fallback for GitHub Actions
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://ec2-34-203-14-116.compute-1.amazonaws.com:5000/")
    MODEL_NAME = "creatorinsight_sentiment_pipeline"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_test_data():
    """Load test data for comparison (handles NaN properly)"""
    try:
        df = pd.read_csv('data/processed/test_processed.csv')
        
        # Handle NaN values
        df['clean_comment'].fillna('', inplace=True)
        
        # Convert to list of strings
        X_test = df['clean_comment'].astype(str).tolist()
        y_test = df['category'].values
        
        # Remove empty entries
        valid = [(x, y) for x, y in zip(X_test, y_test) if x.strip()]
        X_test = [x for x, y in valid]
        y_test = [y for x, y in valid]
        
        print(f"📊 Loaded test data: {len(X_test)} valid samples\n")
        return X_test, y_test
        
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_model(model, X_test, y_test, model_label):
    """Evaluate model (with proper error handling)"""
    try:
        print(f"🔄 Evaluating {model_label}...")
        
        # Predict
        y_pred = model.predict(X_test)
        
        print(f"   ✅ Predictions complete ({len(y_pred)} samples)")
        
        # Calculate metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        
        # Per-class F1
        report = classification_report(y_test, y_pred, output_dict=True)
        
        f1_neg = report.get('-1', {}).get('f1-score', 0)
        f1_neu = report.get('0', {}).get('f1-score', 0)  
        f1_pos = report.get('1', {}).get('f1-score', 0)
        
        metrics = {
            "f1_macro": f1_macro,
            "accuracy": accuracy,
            "f1_negative": f1_neg,
            "f1_neutral": f1_neu,
            "f1_positive": f1_pos
        }
        
        # Print
        print(f"\n{'='*60}")
        print(f"{model_label}")
        print(f"{'='*60}")
        print(f"   F1 Score (Macro):  {metrics['f1_macro']:.4f}")
        print(f"   Accuracy:          {metrics['accuracy']:.4f}")
        print(f"   F1 Negative:       {metrics['f1_negative']:.4f}")
        print(f"   F1 Neutral:        {metrics['f1_neutral']:.4f}")
        print(f"   F1 Positive:       {metrics['f1_positive']:.4f}")
        print()
        
        return metrics
        
    except Exception as e:
        print(f"\n❌ Evaluation failed for {model_label}")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    print("\n" + "="*70)
    print("MODEL COMPARISON REPORT")
    print("="*70 + "\n")
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # ===== LOAD STAGING MODEL (NEWLY TRAINED) =====
    try:
        print("📦 Loading STAGING model (newly trained)...")
        staging_uri = f"models:/{MODEL_NAME}/Staging"
        staging_model = mlflow.sklearn.load_model(staging_uri)
        print("   ✅ Staging model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load staging model: {e}")
        print("   Make sure a model exists in Staging stage")
        sys.exit(1)
    
    # ===== LOAD PRODUCTION MODEL (CURRENT) =====
    try:
        print("🚀 Loading PRODUCTION model (current)...")
        prod_uri = f"models:/{MODEL_NAME}/Production"
        prod_model = mlflow.sklearn.load_model(prod_uri)
        print("   ✅ Production model loaded\n")
        has_production = True
    except Exception as e:
        print(f"ℹ️  No production model found (first deployment?)")
        print(f"   {e}")
        print(f"   Staging model will be promoted automatically\n")
        has_production = False
    
    # ===== EVALUATE STAGING MODEL =====
    staging_metrics = evaluate_model(staging_model, X_test, y_test, "📦 STAGING MODEL")
    
    # ===== EVALUATE PRODUCTION MODEL (if exists) =====
    if has_production:
        prod_metrics = evaluate_model(prod_model, X_test, y_test, "🚀 PRODUCTION MODEL")
        
        # ===== COMPARISON =====
        print("="*70)
        print("COMPARISON")
        print("="*70)
        
        f1_improvement = staging_metrics['f1_macro'] - prod_metrics['f1_macro']
        acc_improvement = staging_metrics['accuracy'] - prod_metrics['accuracy']
        
        print(f"F1 Score Improvement:  {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
        print(f"Accuracy Improvement:  {acc_improvement:+.4f} ({acc_improvement*100:+.2f}%)")
        print()
        
        # ===== DECISION LOGIC =====
        IMPROVEMENT_THRESHOLD = 0.02  # 2% minimum improvement
        
        print("="*70)
        print("DECISION")
        print("="*70)
        
        if f1_improvement >= IMPROVEMENT_THRESHOLD:
            print(f"✅ PROMOTE NEW MODEL")
            print(f"   Reason: F1 improvement of {f1_improvement*100:.2f}% exceeds {IMPROVEMENT_THRESHOLD*100}% threshold")
            print(f"   Staging: {staging_metrics['f1_macro']:.4f}")
            print(f"   Production: {prod_metrics['f1_macro']:.4f}")
            print()
            sys.exit(0)  # Success - promote
            
        elif f1_improvement >= 0:
            print(f"⚠️  MINOR IMPROVEMENT")
            print(f"   F1 improvement: {f1_improvement*100:.2f}% (below {IMPROVEMENT_THRESHOLD*100}% threshold)")
            print(f"   Recommendation: Promote anyway (no degradation)")
            print()
            sys.exit(0)  # Success - promote
            
        else:
            print(f"❌ REJECT NEW MODEL")
            print(f"   Reason: New model performs WORSE")
            print(f"   F1 decline: {f1_improvement*100:.2f}%")
            print(f"   Keep current production model")
            print()
            sys.exit(1)  # Failure - don't promote
    
    else:
        # No production model exists - first deployment
        print("="*70)
        print("DECISION")
        print("="*70)
        print("✅ PROMOTE TO PRODUCTION")
        print("   Reason: No existing production model (first deployment)")
        print(f"   Staging F1: {staging_metrics['f1_macro']:.4f}")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()