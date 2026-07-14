# 🏥 Diabetes Prediction — Random Forest Project

---

## 📁 Project Files

```
diabetes_project/
│
├── backend.py          ← ML brain: download data, train model, evaluate, plot
├── app.py              ← Streamlit UI  (coming next!)
├──  .env
├── requirements.txt    ← All Python libraries needed
├── .env.example        ← Example env file for Kaggle credentials
└── README.md           ← This guide
```

---

## 🚀 Getting Started (Do These Steps IN ORDER)

---

### ✅ STEP 1 — Install Python & VSCode

**Python:**
1. Download from https://python.org/downloads
2. During install → ✅ check **"Add Python to PATH"**

**VSCode:**
1. Download from https://code.visualstudio.com
2. Install the **Python** extension (by Microsoft) from the Extensions panel

---

### ✅ STEP 2 — Open the Project in VSCode

1. File → **Open Folder**
2. Select the `diabetes_project` folder
3. You see `backend.py`, `app.py`, `requirements.txt` in the sidebar ✅

Open the terminal: press **Ctrl + `** (backtick key, top-left of keyboard)

---

### ✅ STEP 3 — Install Libraries

## create the virtual environment
python -m venv venv
venv\Scripts\activate


In the terminal, type:
```bash
pip install -r requirements.txt
```
Wait for the green "Successfully installed…" message. ✅

---

### ✅ STEP 4 — Get Your Kaggle API Key

The dataset lives on Kaggle. You need a free API key to download it.

**Step-by-step:**
1. Go to **https://www.kaggle.com** and log in (or create a free account)
2. Click your **profile picture** (top-right corner)
3. Click **Settings**
4. Scroll down to the **"API"** section
5. Click **"Create New Token"**
6. A file called `kaggle.json` will download automatically
7. Open it in Notepad — it looks like this:
   ```json
   {"username": "yourname", "key": "abc123def456..."}
   ```
8. Keep it handy for the next step!

---

### ✅ STEP 5 — Save Your API Key

This saves your key to a local `.env` file in the project folder.
The backend reads that file and also creates `~/.kaggle/kaggle.json`
automatically when needed.

---

### ✅ STEP 6 — Train and Save the Model Once

```bash
python backend.py
```

You should see:

```
[STEP 1]  Download dataset from Kaggle
✅ Dataset folder: ~/.cache/kagglehub/...
📄 CSV file: .../diabetes_prediction_dataset.csv

[STEP 2]  Load CSV into DataFrame
   gender  age  ...  diabetes
0  Female  80.0 ...  0
...

[STEP 3]  Data Summary
   Rows        : 100,000
   Diabetic    : 8,500   (8.5%)
   Non-Diabetic: 91,500  (91.5%)

[STEP 4]  Preprocess ...
[STEP 5]  Train/Test Split ...
[STEP 6]  Train Random Forest ...
[STEP 6]  Save trained model ...
[STEP 7]  Evaluate
   Test  Accuracy : 97.1%
   Train Accuracy : 99.2%
   ROC-AUC Score  : 98.4%

✅  ALL STEPS PASSED — backend.py works correctly!
💾  Model saved to models/diabetes_model.joblib
👉  Now run:  streamlit run app.py
```

---

### ✅ STEP 7 — Run the Full App

```bash
streamlit run app.py
```

Browser opens at **http://localhost:8501** 🎉

The Streamlit app now loads the saved model file instead of retraining
for every query. If the model file is missing, it will train once and save it.

---

## 📊 Dataset Details

| Column | Type | Meaning |
|--------|------|---------|
| `gender` | Text | Male / Female |
| `age` | Number | 0–80 years |
| `hypertension` | 0 or 1 | Has high blood pressure? |
| `heart_disease` | 0 or 1 | Has heart disease? |
| `smoking_history` | Text | never / current / former / etc. |
| `bmi` | Number | Body Mass Index |
| `HbA1c_level` | Number | Average blood sugar (last 3 months) |
| `blood_glucose_level` | Number | Current blood sugar level |
| `diabetes` | 0 or 1 | **TARGET** — has diabetes? |

---

## ❓ Common Problems & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: kagglehub` | Run `pip install -r requirements.txt` |
| `401 - Unauthorized` on download | Re-run `python setup_kaggle.py` with correct key |
| `No CSV found` error | Delete `~/.cache/kagglehub/` folder and re-run |
| `ModuleNotFoundError: No module named 'backend'` | Make sure `app.py` and `backend.py` are in the same folder |
| Port in use (Streamlit) | Run `streamlit run app.py --server.port 8502` |

---

## 🧪 Experiments for Students

| Experiment | What to change | What to observe |
|------------|---------------|-----------------|
| More trees | n_estimators 10 → 200 | Accuracy vs training time |
| Depth limit | max_depth 3 → 20 | Overfitting gap changes |
| Less data | test_size 10% → 40% | Does accuracy drop? |
| Feature importance | Check the bar chart | HbA1c vs glucose — which matters more? |

---

*Built with Python · kagglehub · scikit-learn · Streamlit · pandas · matplotlib · seaborn*
