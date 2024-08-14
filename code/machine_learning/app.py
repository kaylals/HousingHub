from flask import Flask, request, jsonify, url_for, redirect
from flask_cors import CORS
from random_forest_forecast import rf_api as mid_term_api
from predict_xgboost import xg_api as long_term_api

app = Flask(__name__)
CORS(app)

app.add_url_rule('/random-forest-forecast', 'mid_term_api', mid_term_api, methods=['POST'])
app.add_url_rule('/xgboost', 'long_term_api', long_term_api, methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    range_days = data.get('range')

    # 根据 range 天数选择合适的模型 API 路由
    if range_days <= 30:
        return mid_term_api()
    elif range_days <= 180:
        return mid_term_api()
    else:
        return long_term_api()

if __name__ == '__main__':
    app.run(debug=True)