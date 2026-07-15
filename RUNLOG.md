# Run Log

**Run 1: Silence-Only Baseline**
* **Change:** Ran the provided `baseline.py`.
* **Score:** 1600 ms delay @ 0.0% interruptions.
* **Conclusion:** Relying purely on silence duration is too slow. We need acoustic features to predict the turn ending before the silence stretches too long.

**Run 2: Initial Random Forest (English Only)**
* **Change:** Created a custom feature extractor using `librosa`. Extracted RMS energy (mean, delta), Pitch/F0 (mean, delta), and 13 MFCCs (mean, variance) from the 1.5 seconds immediately preceding the pause. Strictly enforced the causality rule (no future audio). Trained a RandomForestClassifier (`n_estimators=100`).
* **Score:** 100 ms delay @ 4.0% interruptions (English).
* **Conclusion:** Massive improvement. Prosodic features (falling pitch, decaying energy) are strong indicators of End-of-Turn. However, training only on English risks overfitting for the hidden evaluation set.

**Run 3: Robust Multilingual Model (English + Hindi)**
* **Change:** Updated `train.py` to iterate through both `--data_root/english` and `--data_root/hindi` folders. Merged the datasets (496 total samples) to prevent overfitting and generalize the model across different languages.
* **Score:** 100 ms delay @ 2.0% interruptions (English) / 100 ms delay @ 0.0% interruptions (Hindi).
* **Conclusion:** Model successfully generalized across both languages while maintaining an aggressive 100 ms response time without exceeding the 5% interruption limit. Ready for production.