from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from prophet.serialize import model_from_json
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  # Enable CORS
@app.route('/')
def home():
    return jsonify({"message": "API is running successfully!"}), 200
# Load models
model_path = os.path.join(os.path.dirname(__file__), "models/event_models.pkl")
with open(model_path, "rb") as f:
    models = pickle.load(f)

@app.route('/predict', methods=['GET'])
def predict():
    event_type = request.args.get('event', default="Weddings")
    year = int(request.args.get('year', default=2025))
    month = int(request.args.get('month', default=1))

    if event_type not in models:
        return jsonify({"error": "Event type not found!"}), 400

    future = pd.DataFrame({'ds': [pd.Timestamp(year, month, 1)]})
    prophet_model = models[event_type]['prophet']
    xgb_model = models[event_type]['xgb']

    # Prophet prediction
    prophet_forecast = prophet_model.predict(future)
    prophet_prediction = prophet_forecast['yhat'].values[0]

    # XGBoost prediction
    xgb_input = pd.DataFrame({'month': [month], 'year': [year]})
    xgb_prediction = xgb_model.predict(xgb_input)[0]

    # Final prediction (combining both)
    final_prediction = (prophet_prediction * 0.6) + (xgb_prediction * 0.4)

    return jsonify({"event": event_type, "year": year, "month": month, "prediction": round(final_prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the Render-assigned port
    app.run(host="0.0.0.0", port=port, debug=True)
