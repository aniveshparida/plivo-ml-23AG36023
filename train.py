import os
import argparse
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from features import extract_features

def main():
    parser = argparse.ArgumentParser(description="Train RandomForest classifier for EOT detection.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing 'english' and 'hindi' subfolders")
    args = parser.parse_args()

    features_list = []
    labels_list = []

    languages = ['english', 'hindi']
    for lang in languages:
        lang_dir = os.path.join(args.data_root, lang)
        labels_path = os.path.join(lang_dir, "labels.csv")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Could not find labels.csv in {lang_dir}")
            
        print(f"Loading labels from {labels_path}...")
        df = pd.read_csv(labels_path)
        
        # Loop through rows with a progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {lang} audio"):
            audio_path = os.path.join(lang_dir, row['audio_file'])
            pause_start = float(row['pause_start'])
            
            # Extract features using our causality-safe function
            feats = extract_features(audio_path, pause_start)
            features_list.append(feats)
            labels_list.append(1 if row['label'] == 'eot' else 0)

    # 4. Collect extracted features into a DataFrame (X)
    X = pd.DataFrame(features_list)

    # 5. Extract label column and convert to binary target (y)
    # 1 for 'eot', 0 for 'hold'
    y = pd.Series(labels_list).values

    print(f"Training RandomForestClassifier on {len(X)} samples with {X.shape[1]} features...")
    # 6. Train a RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 7. Save the trained model
    model_output_path = "rf_model.pkl"
    joblib.dump(clf, model_output_path)
    print(f"Success! Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    main()
