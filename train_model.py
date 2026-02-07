import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
import os

# 1. Load Data
print("Loading data...")
try:
    mf1 = pd.read_csv('/Users/singhanyadav/Desktop/DATATHON/transactions_datathon_ready.csv')
    mf2 = pd.read_csv('/Users/singhanyadav/Desktop/DATATHON/train.csv')
except FileNotFoundError:
    print("Error: Files not found. Please ensure CSVs are in /Users/singhanyadav/Desktop/DATATHON/")
    exit(1)

# 2. Preprocessing & Feature Engineering
print("Aggregating transaction data...")
mf1_agg = (
    mf1
    .groupby("user_id")
    .agg(
        txn_count=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        std_amount=("amount", "std"),
        online_txn_count=("use_chip", lambda x: (x == "Online Transaction").sum()),
        swipe_txn_count=("use_chip", lambda x: (x == "Swipe Transaction").sum()),
    )
    .reset_index()
)

print("Merging data...")
mf2_fixed = mf2.rename(columns={"id": "user_id"})
df = mf2_fixed.merge(mf1_agg, on="user_id", how="left")

# Handing missing values if any (though notebook said none, merge might introduce them if users have no transactions)
# For this reproduction, we assume data integrity as per notebook
df = df.fillna(0) # Simple fill for aggregations if missing

# 3. Definitions
num_vars = [
    "txn_count", "total_amount", "avg_amount", "max_amount", "std_amount",
    "online_txn_count", "swipe_txn_count", "person_age", "person_income",
    "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length"
]

cat_vars = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
]

selected_features = [
    'loan_grade', 'loan_int_rate', 'loan_percent_income', 'person_home_ownership',
    'person_income', 'cb_person_default_on_file', 'std_amount', 'max_amount',
    'avg_amount', 'total_amount', 'loan_amnt', 'person_emp_length'
]

target = "loan_status"

def woe_iv(data, feature, target):
    eps = 1e-6
    grouped = data.groupby(feature)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]
    grouped["dist_good"] = grouped["good"] / grouped["good"].sum()
    grouped["dist_bad"] = grouped["bad"] / grouped["bad"].sum()
    grouped["woe"] = np.log((grouped["dist_good"] + eps) / (grouped["dist_bad"] + eps))
    grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    return grouped.reset_index(), grouped["iv"].sum()

# Custom binning for loan_percent_income
def loan_percent_income_bin(x):
    if x <= 0.2: return "Low"
    elif x <= 0.35: return "Medium"
    elif x <= 0.6: return "High"
    else: return "Very High"

# 4. Generate WOE Maps and Transform Data
print("Generating WOE maps...")
woe_assets = {} # To store maps and bin boundaries
woe_df = pd.DataFrame({target: df[target]})

for col in selected_features:
    feature_meta = {"name": col}
    
    if col == "loan_percent_income":
        # Custom Coarse Classing
        feature_name = col + "_grp"
        df[feature_name] = df[col].apply(loan_percent_income_bin)
        woe_table, _ = woe_iv(df, feature_name, target)
        
        # Store metadata
        feature_meta["type"] = "custom_bin"
        feature_meta["map"] = dict(zip(woe_table[feature_name], woe_table["woe"]))
        
        # Transform
        woe_df[col + "_WOE"] = df[feature_name].map(feature_meta["map"])
        
    elif col in cat_vars:
        # Categorical
        feature_name = col
        woe_table, _ = woe_iv(df, feature_name, target)
        
        feature_meta["type"] = "categorical"
        feature_meta["map"] = dict(zip(woe_table[feature_name], woe_table["woe"]))
        
        woe_df[col + "_WOE"] = df[feature_name].map(feature_meta["map"])
        
    else:
        # Numerical -> QCut
        feature_name = col + "_bin"
        # Use retbins=True to get boundaries
        df[feature_name], bins = pd.qcut(df[col], q=5, duplicates="drop", retbins=True)
        # Convert intervals to string for JSON serialization consistency if needed, 
        # but for pickle we can keep Intervals. However, app needs to match values to these intervals.
        # We will store the BINS (boundaries) so app can pd.cut new values.
        
        woe_table, _ = woe_iv(df, feature_name, target)
        
        feature_meta["type"] = "numerical_bin"
        feature_meta["bins"] = bins # Array of boundaries
        # Map keys are Interval objects. 
        feature_meta["map"] = dict(zip(woe_table[feature_name], woe_table["woe"]))
        
        woe_df[col + "_WOE"] = df[feature_name].map(feature_meta["map"])
    
    woe_assets[col] = feature_meta

# 5. Model Training
print("Training model...")
X = woe_df.drop(columns=[target])
y = woe_df[target]

# Create output directory
os.makedirs("model_assets", exist_ok=True)

model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
model.fit(X, y)

# 6. Scaling Parameters
BASE_SCORE = 600
PDO = 50
BASE_ODDS = 20

factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

params = {
    "factor": factor,
    "offset": offset,
    "cutoff_score": 580,
    "selected_features": selected_features
}

# 7. Save Artifacts
print("Saving artifacts...")
joblib.dump(model, "model_assets/model.pkl")
joblib.dump(woe_assets, "model_assets/woe_assets.pkl")

# Convert numpy types to python native for JSON
params_serializable = {k: float(v) if isinstance(v, (np.float64, np.float32)) else v for k, v in params.items()}
with open("model_assets/params.json", "w") as f:
    json.dump(params_serializable, f, indent=4)

print("Done! Artifacts saved in 'model_assets/'.")
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficients: {model.coef_[0]}")
print(f"WOE Assets keys: {list(woe_assets.keys())}")
