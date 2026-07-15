import os
import argparse
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from features import extract_features
from score import evaluate, THRESHOLDS, DELAYS, TIMEOUT_S


def build_dataset(data_root, languages=("english", "hindi")):
    """Extract features once for every pause across all languages.

    Returns X (features), y (0/1 labels), groups (turn-level id used to keep
    every pause of a turn on the same side of any CV split), and meta (the
    per-row info score.py's policy needs: turn_id, pause duration, label).
    """
    features_list, labels_list, groups, meta = [], [], [], []
    for lang in languages:
        lang_dir = os.path.join(data_root, lang)
        labels_path = os.path.join(lang_dir, "labels.csv")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Could not find labels.csv in {lang_dir}")

        print(f"Loading labels from {labels_path}...")
        df = pd.read_csv(labels_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {lang} audio"):
            audio_path = os.path.join(lang_dir, row["audio_file"])
            pause_start = float(row["pause_start"])

            # Extract features using our causality-safe function
            feats = extract_features(audio_path, pause_start)
            features_list.append(feats)
            labels_list.append(1 if row["label"] == "eot" else 0)
            # Prefix with language so an english turn_id can never collide
            # with a hindi turn_id of the same name in GroupKFold.
            groups.append(f"{lang}:{row['turn_id']}")
            meta.append({
                "lang": lang,
                "turn_id": row["turn_id"],
                "dur": float(row["pause_end"]) - float(row["pause_start"]),
                "label": row["label"],
            })

    X = pd.DataFrame(features_list)
    y = np.array(labels_list)
    return X, y, groups, meta


def simulate_score(meta_subset, probs_subset, budget=0.05):
    """Re-run score.py's exact sweep/policy logic on held-out predictions.

    This gives the same "mean response delay @ <=5% interrupted turns"
    number score.py would report, but computed on predictions the model
    never trained on -- an honest stand-in for the hidden test set.
    """
    pauses = [
        {"turn_id": m["turn_id"], "dur": m["dur"], "label": m["label"], "p": p}
        for m, p in zip(meta_subset, probs_subset)
    ]
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d)
            if cut <= budget and (best is None or lat < best["latency"]):
                best = {"latency": lat, "cutoff": cut, "threshold": t, "delay": d}
    if best is None:
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    return best


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest classifier for EOT detection.")
    parser.add_argument("--data_root", type=str, required=True,
                         help="Root directory containing 'english' and 'hindi' subfolders")
    parser.add_argument("--n_splits", type=int, default=5,
                         help="Grouped CV folds for an honest, held-out validation estimate "
                              "(grouped by turn_id so no turn's pauses leak across train/val)")
    args = parser.parse_args()

    X, y, groups, meta = build_dataset(args.data_root)
    n_turns = len(set(groups))
    print(f"\nLoaded {len(X)} pause samples ({X.shape[1]} features) across {n_turns} turns.")

    # ---------------------------------------------------------------
    # Honest validation: GroupKFold means every pause from a given turn
    # stays entirely in train OR entirely in validation for that fold --
    # a plain random split would leak correlated pauses from the same
    # recording across both sides and inflate the score, same as scoring
    # in-sample does.
    # ---------------------------------------------------------------
    print(f"\nRunning {args.n_splits}-fold grouped cross-validation (grouped by turn_id)...")
    gkf = GroupKFold(n_splits=args.n_splits)
    oof_probs = np.zeros(len(y))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        fold_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        fold_clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof_probs[va_idx] = fold_clf.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  fold {fold + 1}/{args.n_splits}: train={len(tr_idx)} val={len(va_idx)}")

    oof_auc = roc_auc_score(y, oof_probs)
    print(f"\nOut-of-fold AUC (both languages combined, held-out): {oof_auc:.3f}")

    print("\nHeld-out (out-of-fold) score.py-equivalent metric per language:")
    for lang in ("english", "hindi"):
        idx = [i for i, m in enumerate(meta) if m["lang"] == lang]
        lang_meta = [meta[i] for i in idx]
        lang_probs = oof_probs[idx]
        res = simulate_score(lang_meta, lang_probs)
        print(f"  [OOF] {lang}: mean delay={res['latency']*1000:.0f} ms  "
              f"interrupted turns={res['cutoff']*100:.1f}%  "
              f"(operating point threshold={res['threshold']}, delay={res['delay']*1000:.0f} ms)")

    # ---------------------------------------------------------------
    # Final model: fit on 100% of the data (no data is wasted -- the
    # validation above only used temporary fold-models, never the
    # shipped one). oob_score gives a second, essentially free
    # generalization signal for this exact final model.
    # ---------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    clf.fit(X, y)
    print(f"\nFinal model (fit on 100% of data) OOB accuracy: {clf.oob_score_:.3f}")

    model_output_path = "rf_model.pkl"
    joblib.dump(clf, model_output_path)
    print(f"Success! Model trained and saved to {model_output_path}")


if __name__ == "__main__":
    main()
