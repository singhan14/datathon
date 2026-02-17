# 📊 Datathon — Credit Scorecard Model

A machine learning project that builds a **logistic regression-based credit scorecard** to predict credit default risk. The model uses Weight of Evidence (WoE) encoding and produces interpretable credit scores, complete with a web app interface for live scoring.

🔗 **Live App:** [https://datathon-zoaejyzqzkmlsft8kfni7j.streamlit.app](https://datathon-zoaejyzqzkmlsft8kfni7j.streamlit.app)

---

## 🗂️ Project Structure

```
datathon/
├── Datathon.ipynb              # Exploratory data analysis & model development notebook
├── train.csv                   # Training dataset
├── train_model.py              # Script to train and serialize the scorecard model
├── app.py                      # Web application for real-time credit scoring
├── requirements.txt            # Python dependencies
├── logistic_scorecard_model.joblib  # Trained logistic regression model
├── model_features.joblib       # Selected model features
├── woe_maps.joblib             # Weight of Evidence transformation maps
├── score_factor.joblib         # Scorecard scaling factor
├── score_offset.joblib         # Scorecard scaling offset
├── score_cutoff.joblib         # Decision cutoff score
└── model_assets/               # Additional model artifacts
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/singhan14/datathon.git
cd datathon

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

To retrain the scorecard model from scratch using the provided dataset:

```bash
python train_model.py
```

This will:
1. Load and preprocess `train.csv`
2. Apply Weight of Evidence (WoE) binning and encoding
3. Train a logistic regression scorecard
4. Save all model artifacts (`.joblib` files) to the project root

---

## 🌐 Running the App

### Live Demo

👉 [https://datathon-zoaejyzqzkmlsft8kfni7j.streamlit.app](https://datathon-zoaejyzqzkmlsft8kfni7j.streamlit.app)

### Run Locally

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

The app allows you to input applicant features and receive a real-time **credit score** along with an **Approve / Reject** decision based on the learned cutoff.

---

## 🧠 Methodology

The model is built using a classic **credit scorecard** approach:

- **Weight of Evidence (WoE)** transforms categorical and numerical features into a standardized format that captures the relationship between each feature bin and the target variable (default).
- **Logistic Regression** is trained on the WoE-encoded features for interpretability and regulatory compliance.
- **Scorecard Scaling** converts raw log-odds into an easy-to-interpret score (typically in the range of 300–850) using a factor and offset.
- A **score cutoff** determines the approve/reject threshold.

---

## 📦 Dependencies

Key libraries used (see `requirements.txt` for full list):

- `scikit-learn` — model training and evaluation
- `pandas` / `numpy` — data manipulation
- `joblib` — model serialization
- `streamlit` — web application

---

## 📓 Notebook

The `Datathon.ipynb` notebook walks through the full pipeline:

1. Data exploration and visualization
2. Feature engineering and WoE binning
3. Model training and evaluation (KS statistic, Gini, AUC)
4. Scorecard generation and cutoff selection

---

## 📄 License

This project was developed for a datathon competition. Feel free to fork, modify, and build upon it.
