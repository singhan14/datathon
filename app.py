import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

# 1. Page Config & Styling
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "God-level" UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1a1a2e;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
    }
    h2, h3 {
        color: #16213e;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        background-color: #0f3460;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1a1a2e;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Load Artifacts
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model_assets/model.pkl")
        woe_assets = joblib.load("model_assets/woe_assets.pkl")
        with open("model_assets/params.json", "r") as f:
            params = json.load(f)
        return model, woe_assets, params
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run `train_model.py` first.")
        st.stop()

model, woe_assets, params = load_assets()

# 3. Helper Functions
def loan_percent_income_bin(x):
    if x <= 0.2: return "Low"
    elif x <= 0.35: return "Medium"
    elif x <= 0.6: return "High"
    else: return "Very High"

def calculate_score(pd_prob):
    # Score = Offset - Factor * ln(odds)
    # odds = p / (1 - p)
    # But here we use the breakdown form: Offset - Factor * log_odds_of_bad
    # Since model predicts P(Bad), log_odds = log(p/1-p) = logit
    # Score = Offset - Factor * (intercept + sum(coef * woe))
    # This is equivalent to Score = Offset - Factor * model.decision_function(X)
    
    # We can reconstruct it from probability to be safe and consistent with params
    odds = pd_prob / (1 - pd_prob + 1e-10)
    score = params["offset"] - params["factor"] * np.log(odds + 1e-10)
    return score

# 4. App Layout
st.title("💳 AI Credit Risk Evaluator")
st.markdown("---")

col1, col2 = st.columns([1, 2])

# Initialize input dict
raw_input = {}

with col1:
    st.subheader("📋 Applicant Details")
    with st.form("input_form"):
        # Iterate features in order to create UI
        for feature in params["selected_features"]:
            meta = woe_assets[feature]
            label = feature.replace("_", " ").title()
            
            if meta["type"] == "numerical_bin":
                bins = meta["bins"]
                min_val = float(bins[0])
                max_val = float(bins[-1])
                # Handle infinite boundaries for UI by capping
                if min_val == -np.inf: min_val = 0.0
                if max_val == np.inf: max_val = 1000000.0 # Reasonable cap
                
                # Check for specific "amount" fields to give better defaults/steps
                step = 1.0
                if "int_rate" in feature or "percent" in feature:
                    step = 0.01
                
                # Use mean or reasonable default
                default_val = (min_val + max_val) / 2
                if default_val > 100000: default_val = 50000.0
                
                raw_input[feature] = st.number_input(
                    label, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=default_val,
                    step=step
                )
                
            elif meta["type"] == "categorical":
                # Keys of map are the categories
                options = list(meta["map"].keys())
                raw_input[feature] = st.selectbox(label, options=options)
                
            elif meta["type"] == "custom_bin":
                # loan_percent_income
                # Range 0.0 to 1.0 normally
                raw_input[feature] = st.slider(
                    label + " (Ratio)", 
                    min_value=0.0, 
                    max_value=1.5, 
                    value=0.15,
                    step=0.01
                )
                
        submit = st.form_submit_button("Analyze Risk Profile")

# 5. Prediction Logic
if submit:
    # Transform raw input to WOE
    woe_vector = []
    
    for feature in params["selected_features"]:
        val = raw_input[feature]
        meta = woe_assets[feature]
        woe_val = 0.0
        
        if meta["type"] == "numerical_bin":
            bins = meta["bins"]
            # pd.cut returns categories, we need to match the Interval key in map
            # Since Interval architecture in simple dicts can be tricky (string vs obj),
            # we do a manual check against bins to be robust
            
            # Find which bin val falls into
            # bin[i] < val <= bin[i+1]
            found = False
            for i in range(len(bins)-1):
                lower = bins[i]
                upper = bins[i+1]
                if i == 0: # First bin includes lower edge if we follow include_lowest=True logic or handle -inf
                    if val >= lower and val <= upper: # Loose check
                        # Construct the interval key that matches the map
                        # The map keys are likely pandas Interval objects if saved via joblib/pickle
                        # We try to match by finding the key in the map that overlaps
                        for k in meta["map"].keys():
                            if k.left == lower and k.right == upper:
                                woe_val = meta["map"][k]
                                found = True
                                break
                else:
                    if val > lower and val <= upper:
                        for k in meta["map"].keys():
                            if k.left == lower and k.right == upper:
                                woe_val = meta["map"][k]
                                found = True
                                break
                if found: break
            
            if not found:
                # Fallback: maybe out of bounds?
                # Check for open intervals
                for k, v in meta["map"].items():
                    if val in k:
                        woe_val = v
                        found = True
                        break
            
        elif meta["type"] == "categorical":
            woe_val = meta["map"].get(val, 0.0) # Default to 0 (neutral) if unseen
            
        elif meta["type"] == "custom_bin":
            category = loan_percent_income_bin(val)
            woe_val = meta["map"].get(category, 0.0)
            
        woe_vector.append(woe_val)

    # Convert to 2D array
    X_input = np.array(woe_vector).reshape(1, -1)
    
    # Predict
    prob_default = model.predict_proba(X_input)[0, 1]
    
    # Calculate Score
    score = calculate_score(prob_default)
    score = int(round(score))
    
    # Decision
    decision = "APPROVE" if score >= params["cutoff_score"] else "REJECT"
    
    # 6. Display Results
    with col2:
        st.write("## Assessment Results")
        
        # Row 1: Key Metrics
        m1, m2, m3 = st.columns(3)
        
        decision_color = "green" if decision == "APPROVE" else "red"
        
        with m1:
            st.markdown(f'<div class="metric-card"><h3>Decision</h3><h2 style="color:{decision_color}">{decision}</h2></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><h3>Credit Score</h3><h2>{score}</h2></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><h3>Default Prob.</h3><h2>{prob_default:.2%}</h2></div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Creditworthiness"},
            gauge = {
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': decision_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [300, 500], 'color': '#ffcccb'},
                    {'range': [500, 650], 'color': '#ffffd1'},
                    {'range': [650, 850], 'color': '#d1ffbd'}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': params["cutoff_score"]}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation (Feature Importance contribution)
        with st.expander("🔍 Risk Factor Analysis"):
            contributions = []
            for i, feature in enumerate(params["selected_features"]):
                coef = model.coef_[0][i]
                woe = woe_vector[i]
                contrib = coef * woe * -params["factor"] # Impact on score
                contributions.append((feature, contrib))
            
            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            st.write("Top drivers for this score:")
            for feat, contrib in contributions[:5]:
                direction = "Positive" if contrib > 0 else "Negative"
                icon = "✅" if contrib > 0 else "⚠️"
                st.write(f"{icon} **{feat.replace('_', ' ').title()}**: {direction} impact ({int(contrib)} pts)")
else:
    with col2:
        st.info("👈 Enter applicant details and click 'Analyze' to generate a credit score.")
        # Static placeholder image or text could go here
