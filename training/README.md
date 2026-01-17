Mind Reader - Training Notes

This folder contains a Kaggle-ready training script for facial expression
recognition. The dataset is used ONLY to train a model; the frontend uses the
exported TF.js files.

Files
- training/kaggle_emotion_training.py: Kaggle notebook-style script.

How to use on Kaggle
1) Create a new Kaggle notebook.
2) Upload the dataset (CSV or image folders).
3) Paste the contents of training/kaggle_emotion_training.py into the notebook.
4) Run all cells to train and save expression_model.h5.

Export to TF.js (run locally)
pip install tensorflowjs
tensorflowjs_converter --input_format=keras expression_model.h5 frontend/public/web_model

Important
- Keep EMOTION_LABELS in frontend/src/App.jsx in the same order used during training.
- The TF.js files must live in frontend/public/web_model/
  (model.json + shard files).
