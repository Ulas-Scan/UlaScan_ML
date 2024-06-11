from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import get_model, get_tokenizer, predict_results
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
from functools import wraps
import os

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

MODEL_NAME = 'cahya/bert-base-indonesian-522M'
MAX_LENGTH = 100
PRETRAINED_PATH = 'transformers-bert'

tokenizer = get_tokenizer(MODEL_NAME)
model = get_model(PRETRAINED_PATH)

# Get API key from environment variable
API_KEY = os.getenv('API_KEY')

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('api-key')
        if key and key == API_KEY:
            return f(*args, **kwargs)
        else:
            abort(401)  # Unauthorized
    return decorated_function

@app.route("/predict", methods=['POST'])
@require_api_key
def predict():
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400
    
    statements = data.get("statements")
    if statements is None:
        return jsonify({"error": "No statements provided"}), 400
    
    # statements: list of reviews
    report = {'Positive': 0, 'Negative': 0}
    
    def process_statements(statements):
        try:
            logits = predict_results(statements, tokenizer, model, MAX_LENGTH)
            return logits
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
    
    predictions = process_statements(statements)
    for pred in predictions:
        if pred[0] > pred[1]:
            report['Negative'] += 1
        else:
            report['Positive'] += 1

    return jsonify(report)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
