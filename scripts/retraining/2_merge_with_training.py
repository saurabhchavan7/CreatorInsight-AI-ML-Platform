"""
DVC Stage: Merge new samples with training data
Runs automatically as part of DVC pipeline
"""

import pandas as pd
from pathlib import Path
import sys
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

NEW_DATA_PATH = "data/retraining/new_samples.csv"
TRAIN_PATH = "data/interim/train.csv"


def preprocess_text(text):
    """Match your preprocessing pipeline"""
    try:
        text = str(text).lower().strip()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)
        
        try:
            sw = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
        except:
            sw = set()
        
        words = [w for w in text.split() if w not in sw]
        text = " ".join(words)
        
        try:
            lemmatizer = WordNetLemmatizer()
            text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
        except:
            pass
        
        return text
    except:
        return str(text)


def main():
    # Check if new data exists
    if not Path(NEW_DATA_PATH).exists():
        print("ℹ️  No new training data - skipping merge")
        print("   (This is normal for regular pipeline runs)\n")
        return 0
    
    print("\n🔄 Merging new training data...\n")
    
    # Load
    train_df = pd.read_csv(TRAIN_PATH)
    new_df = pd.read_csv(NEW_DATA_PATH)
    
    print(f"   Original: {len(train_df)} rows")
    print(f"   New data: {len(new_df)} rows")
    
    # Preprocess new data
    if 'text' in new_df.columns:
        new_df['clean_comment'] = new_df['text'].apply(preprocess_text)
        new_df = new_df[['clean_comment', 'category']]
    
    # Merge
    combined = pd.concat([train_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=['clean_comment'], keep='first', inplace=True)
    
    print(f"   Merged: {len(combined)} rows")
    print(f"   Net new: {len(combined) - len(train_df)}\n")
    
    # Save
    combined.to_csv(TRAIN_PATH, index=False)
    
    print(f"✅ Merge complete: {len(combined)} total examples\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())