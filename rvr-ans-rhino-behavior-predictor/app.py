import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

# --- Configuration ---
app = Flask(__name__)
# Define the path to where the model/scalers are saved
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global Variables for Model & Scalers ---
# These are loaded once when the app starts
try:
    # 1. Load the Keras Model (ANN)
    MODEL_PATH = os.path.join(MODEL_DIR, 'rhino_interaction_ann_model.keras')
    model = load_model(MODEL_PATH)

    # 2. Load Scalers
    with open(os.path.join(MODEL_DIR, 'rhino_scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'rhino_scaler_Y.pkl'), 'rb') as f:
        scaler_Y = pickle.load(f)

    # 3. Load Feature Statistics (for min/max constraints and mean for scenarios)
    with open(os.path.join(MODEL_DIR, 'rhino_feature_stats.pkl'), 'rb') as f:
        FEATURE_STATS = pickle.load(f)
    
    # 4. Get the list of all 64 feature names from the scaler
    ALL_FEATURE_NAMES = scaler_X.feature_names_in_

except Exception as e:
    print(f"Error loading model or scalers: {e}")
    model = None
    # Use empty lists/dicts as fallback if loading fails
    ALL_FEATURE_NAMES = []
    FEATURE_STATS = {'mean': {}, 'min': {}, 'max': {}}


# --- Feature Selection: Choose 8 most important/representative features ---
# Based on common rhino behavior and potential impact on Human Interaction
# The chosen features are the most relevant non-target columns from the 64 available.
INPUT_FEATURE_NAMES = [
    'Duration_Minutes_Resting',
    'Frequency_Foraging',
    'Frequency_Roaming (outdoor enclosure)',
    'Morning_Foraging',
    'Morning_Resting',
    'Evening_Roaming (indoor enclosure)',
    'Frequency_Provisioned Feeding',
    'Duration_Minutes_Standing Still'
]

# --- Helper Function for Prediction ---
def make_prediction(input_data_df):
    """Fills the 64 features and makes a prediction."""
    
    # 1. Create a base dataframe with all 64 features initialized to their mean
    X_mean = pd.Series(FEATURE_STATS['mean'], index=ALL_FEATURE_NAMES)
    X_full = pd.DataFrame([X_mean], columns=ALL_FEATURE_NAMES)
    
    # 2. Update the 8 specific input features with user/scenario data
    for col in input_data_df.columns:
        if col in X_full.columns:
            X_full[col] = input_data_df[col].iloc[0]
            
    # 3. Scale the full 64-feature input
    X_scaled = scaler_X.transform(X_full)

    # 4. Predict
    prediction_scaled = model.predict(X_scaled, verbose=0)
    
    # 5. Inverse transform and apply post-processing (rounding/non-negativity)
    prediction_corrected = np.maximum(0, prediction_scaled)
    prediction = scaler_Y.inverse_transform(prediction_corrected)
    
    prediction_df = pd.DataFrame(prediction, columns=scaler_Y.feature_names_in_)
    
    # Apply rounding to Frequency/Time columns and 2 decimal places for Duration
    frequency_cols = [col for col in prediction_df.columns if 'Frequency' in col or 'Morning' in col or 'Evening' in col]
    duration_cols = [col for col in prediction_df.columns if 'Duration' in col]
    
    prediction_df[frequency_cols] = prediction_df[frequency_cols].round(0).astype(int)
    prediction_df[duration_cols] = prediction_df[duration_cols].round(2)
    
    # Extract the results for easier display
    results = {
        'VDA_Freq': prediction_df['Frequency_Visitor-Directed Approach'].iloc[0],
        'VDA_Duration': prediction_df['Duration_Minutes_Visitor-Directed Approach'].iloc[0],
        'LKR_Freq': prediction_df['Frequency_Lack of Response to Visitors'].iloc[0],
        'LKR_Duration': prediction_df['Duration_Minutes_Lack of Response to Visitors'].iloc[0],
    }
    
    return results

# --- Prediction Function for Scenarios (Static Data from Notebook) ---
def get_scenario_predictions():
    """Returns the pre-calculated scenario results."""
    # These results are hardcoded from the notebook analysis for consistency
    return [
        {
            'name': 'Scenario A (High Rest/Low Activity)',
            'vda_freq': 7,
            'insight': "Highest predicted interaction frequency. Suggests heightened sensitivity to visitors during prolonged rest."
        },
        {
            'name': 'Scenario B (High Active Morning)',
            'vda_freq': 6,
            'insight': "Moderate interaction. Rhino is highly active and roaming, potentially distracted from visitors."
        },
        {
            'name': 'Scenario C (Average Baseline)',
            'vda_freq': 6,
            'insight': "Baseline prediction based on average daily activity."
        }
    ]

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return render_template('index.html', error="Model files could not be loaded. Check 'app.py' console for details.")

    # Prepare data for rendering input form
    input_fields_data = []
    for name in INPUT_FEATURE_NAMES:
        # Generate user-friendly labels (e.g., 'Duration_Minutes_Resting' -> 'Resting Duration (Min)')
        parts = name.split('_')
        if len(parts) > 1 and parts[0] in ['Duration', 'Frequency', 'Morning', 'Evening']:
            label = f"{parts[-1]} {parts[0].replace('Duration', 'Duration (Min)').replace('Frequency', 'Frequency (Count)')}"
        else:
            label = name.replace('_', ' ').title()
            
        input_fields_data.append({
            'id': name,
            'label': label,
            'min': int(FEATURE_STATS['min'].get(name, 0)),
            'max': int(FEATURE_STATS['max'].get(name, 100)),
            'mean': round(FEATURE_STATS['mean'].get(name, 0)),
            'value': round(FEATURE_STATS['mean'].get(name, 0)) # Default to mean
        })

    prediction_results = None
    
    if request.method == 'POST':
        # 1. Collect user input from the form
        user_input = {}
        for field in INPUT_FEATURE_NAMES:
            try:
                # Convert all inputs to float as expected by the model scaler
                user_input[field] = [float(request.form.get(field))]
            except (ValueError, TypeError):
                # Fallback to mean if input is invalid
                user_input[field] = [FEATURE_STATS['mean'].get(field, 0)]
                
        input_df = pd.DataFrame(user_input)
        
        # 2. Make the prediction
        prediction_results = make_prediction(input_df)
        
    scenario_results = get_scenario_predictions()

    return render_template(
        'index.html',
        fields=input_fields_data,
        prediction=prediction_results,
        scenarios=scenario_results
    )

if __name__ == '__main__':
    # Flask runs on 127.0.0.1:5000 by default
    app.run(debug=True, use_reloader=False)