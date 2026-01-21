"""
Step 1 of retraining: Download S3 predictions and select best examples
This runs BEFORE DVC pipeline
"""

import json
import pandas as pd
import boto3
import sys
from pathlib import Path

S3_BUCKET = "creator-insight-dvc-bucket"
S3_PREFIX = "retraining_data/"
OUTPUT_PATH = "data/retraining/new_samples.csv"


def download_predictions():
    """Download all saved predictions from S3"""
    print("\n📥 Downloading predictions from S3...")
    
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter='/')
    
    if 'Contents' not in response:
        print("   No predictions found in S3")
        return []
    
    all_preds = []
    file_count = 0
    
    for obj in response['Contents']:
        key = obj['Key']
        
        # Skip folders and archive
        if key.endswith('/') or 'archive/' in key:
            continue
        
        if not key.endswith('.json'):
            continue
        
        try:
            file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            content = json.loads(file_obj['Body'].read())
            all_preds.extend(content['data'])
            file_count += 1
            print(f"   ✅ {key.split('/')[-1]}: {len(content['data'])} predictions")
        except Exception as e:
            print(f"   ⚠️  Skipped {key}: {e}")
    
    print(f"\n📊 Total: {len(all_preds)} predictions from {file_count} files\n")
    return all_preds, file_count


def select_best_examples(predictions, target=100):
    """Enterprise 3-tier selection"""
    
    if len(predictions) < 50:
        print(f"⚠️  Only {len(predictions)} predictions - need at least 50")
        return []
    
    print("🎯 Applying 3-tier selection strategy...\n")
    
    selected = []
    
    # Tier 1: Hard negatives (30%)
    uncertain = [p for p in predictions if 0.4 < p.get('confidence', 1) < 0.65]
    uncertain.sort(key=lambda x: x.get('confidence', 1))
    tier1 = uncertain[:30]
    selected.extend(tier1)
    print(f"   Tier 1 (Hard negatives): {len(tier1)}")
    
    # Tier 2: Stratified confident (50%)
    confident = [p for p in predictions if p.get('confidence', 0) > 0.90]
    by_sent = {
        '-1': [p for p in confident if str(p['sentiment']) == '-1'],
        '0': [p for p in confident if str(p['sentiment']) == '0'],
        '1': [p for p in confident if str(p['sentiment']) == '1']
    }
    
    for sent, items in by_sent.items():
        selected.extend(items[:17])
    
    tier2_count = len(selected) - len(tier1)
    print(f"   Tier 2 (Stratified): {tier2_count}")
    
    # Tier 3: Diverse (20%)
    remaining = [p for p in predictions if p not in selected]
    
    def diversity(text):
        words = text.lower().split()
        return len(set(words)) / (len(words) + 1) if words else 0
    
    scored = sorted(remaining, key=lambda p: diversity(p.get('text', '')), reverse=True)
    tier3 = scored[:20]
    selected.extend(tier3)
    print(f"   Tier 3 (Diverse): {len(tier3)}\n")
    
    return selected[:target]


def save_selected_data(selected):
    """Save selected examples to CSV"""
    
    # Create directory
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'text': p.get('text') or p.get('comment', ''),
            'category': int(p['sentiment'])
        }
        for p in selected
    ])
    
    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} selected examples to: {OUTPUT_PATH}\n")
    
    return OUTPUT_PATH


def archive_s3_files():
    """Move processed S3 files to archive/"""
    print("📦 Archiving S3 files...\n")
    
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX, Delimiter='/')
    
    if 'Contents' not in response:
        print("   No files to archive\n")
        return
    
    archived = 0
    for obj in response['Contents']:
        key = obj['Key']
        
        if key.endswith('/') or 'archive/' in key or not key.endswith('.json'):
            continue
        
        # Move to archive
        new_key = key.replace('retraining_data/', 'retraining_data/archive/')
        
        s3.copy_object(
            Bucket=S3_BUCKET,
            CopySource={'Bucket': S3_BUCKET, 'Key': key},
            Key=new_key
        )
        
        s3.delete_object(Bucket=S3_BUCKET, Key=key)
        
        print(f"   ✅ Archived: {key.split('/')[-1]}")
        archived += 1
    
    print(f"\n   Total: {archived} files archived\n")


def main():
    print("\n" + "="*70)
    print("STEP 1: DOWNLOAD & SELECT RETRAINING DATA")
    print("="*70 + "\n")
    
    # Download
    preds, file_count = download_predictions()
    
    if len(preds) < 50:
        print("❌ Insufficient data for retraining")
        print(f"   Found: {len(preds)} predictions")
        print(f"   Need: At least 50 (preferably 100+)")
        print("\n💡 Collect more data by analyzing videos and clicking 'Save for Retraining'\n")
        return 1
    
    # Select
    selected = select_best_examples(preds, target=100)
    
    if len(selected) < 50:
        print(f"❌ Selection yielded only {len(selected)} examples (need 50+)")
        return 1
    
    # Save
    output_file = save_selected_data(selected)
    
    # Archive S3
    archive_s3_files()
    
    print("="*70)
    print("✅ STEP 1 COMPLETE")
    print("="*70 + "\n")
    
    print(f"📝 Selected data saved: {output_file}")
    print(f"📝 S3 files archived")
    print(f"📊 Ready for DVC pipeline with {len(selected)} new examples\n")
    
    print("🚀 Next: Run 'dvc repro' to retrain model\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)