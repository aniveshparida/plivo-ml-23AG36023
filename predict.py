import os
import argparse
import pandas as pd
import joblib
from tqdm import tqdm

from features import extract_features

def main():
    parser = argparse.ArgumentParser(description="Predict EOT probabilities using a trained RandomForest model.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing labels.csv and audio files")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Output path for the predictions CSV file")
    args = parser.parse_args()

    # 2. Load the trained model
    model_path = "rf_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first.")
    
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # 3. Load labels.csv
    labels_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find labels.csv in {args.data_dir}")

    print(f"Loading data from {labels_path}...")
    df = pd.read_csv(labels_path)
    
    required_cols = {'turn_id', 'pause_index', 'audio_file', 'pause_start'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"labels.csv is missing required columns. Must contain: {required_cols}")

    results = []

    print("Generating predictions...")
    # 4. Loop through each row in the dataframe using tqdm
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features & predicting"):
        turn_id = row['turn_id']
        pause_index = row['pause_index']
        audio_path = os.path.join(args.data_dir, row['audio_file'])
        pause_start = float(row['pause_start'])
        
        # 5. Extract features
        feats_dict = extract_features(audio_path, pause_start)
        
        # 6. Convert to DataFrame (single row)
        # Note: the features must match the exact columns the model was trained on.
        # Since we use extract_features in both train and predict and put them in a DataFrame,
        # column names and order will match naturally.
        X = pd.DataFrame([feats_dict])
        
        # 7. Get probability of 'eot' (class 1)
        # model.classes_ will be [0, 1] based on train.py mapping.
        # Index 1 corresponds to '1' which is 'eot'
        p_eot = model.predict_proba(X)[0][1]
        
        # 8. Store results
        results.append({
            'turn_id': turn_id,
            'pause_index': pause_index,
            'p_eot': p_eot
        })

    # 9. Save a CSV with EXACTLY the three columns
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False, columns=['turn_id', 'pause_index', 'p_eot'])
    
    print(f"Success! Saved predictions to {args.out}")

if __name__ == "__main__":
    main()
