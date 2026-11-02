"""
CreditPulse — web app for the credit-card default model.

Run from the repository root:
    python app/app.py            -> http://127.0.0.1:5000
Production (Render, Railway...):
    gunicorn app.app:app
"""
import bisect
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # makes src.features importable for joblib
from src.features import add_features  # noqa: E402

MODEL = joblib.load(ROOT / 'models' / 'credit_default_model.joblib')
CARD = json.loads((ROOT / 'models' / 'model_card.json').read_text())
INPUTS = CARD['input_columns']
# Validation clients' scores, riskiest first: the "portfolio" a new client is ranked in
PORTFOLIO = json.loads((ROOT / 'models' / 'portfolio_scores.json').read_text())['scores']
PORTFOLIO_ASC = sorted(PORTFOLIO)

MONTHS = ['September', 'August', 'July', 'June', 'May', 'April']  # index 0 = latest
STATUS_FIELDS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

app = Flask(__name__)


def read_payload(data):
    """Validate the JSON sent by the form and return a one-row DataFrame."""
    errors, row = [], {}
    for col in INPUTS:
        value = data.get(col)
        try:
            row[col] = float(value)
        except (TypeError, ValueError):
            errors.append(col)
    if errors:
        return None, f'Fill in every field. Missing or not a number: {", ".join(errors)}'
    if row['LIMIT_BAL'] <= 0:
        return None, 'The credit limit must be greater than 0.'
    if any(row[c] < 0 for c in INPUTS if c.startswith('PAY_AMT')):
        return None, 'Payment amounts cannot be negative.'
    for c in STATUS_FIELDS:
        if not -2 <= row[c] <= 8:
            return None, 'Repayment status must be between -2 and 8.'
    return pd.DataFrame([row]), None


def explain(df):
    """Plain-language reasons based on the engineered indicators."""
    f = add_features(df).iloc[0]
    med = CARD['portfolio_medians']
    reasons = []
    latest = int(f['PAY_0'])
    if latest >= 2:
        reasons.append(('up', f'{latest} months behind on the September bill. '
                              'In the data, about 7 in 10 such clients defaulted.'))
    elif latest == 1:
        reasons.append(('up', 'One month late on the September bill. '
                              'About 1 in 3 such clients defaulted.'))
    elif latest == -1:
        reasons.append(('down', 'Paid the September bill in full.'))

    late = int(f['MONTHS_LATE'])
    if late >= 2:
        reasons.append(('up', f'Late in {late} of the last 6 months.'))
    elif late == 0:
        reasons.append(('down', 'No late payment in the last 6 months.'))

    util = f['UTIL_LAST']
    if util >= 0.8:
        reasons.append(('up', f'Uses {util:.0%} of the credit limit '
                              f'(typical client: {med["UTIL_LAST"]:.0%}).'))
    elif util <= 0.3:
        reasons.append(('down', f'Uses only {max(util, 0):.0%} of the credit limit.'))

    ratio = f['PAY_RATIO']
    if ratio < 0.10:
        reasons.append(('up', f'Repaid only {ratio:.0%} of the previous bills.'))
    elif ratio >= 0.9:
        reasons.append(('down', 'Repaid almost all of the previous bills.'))
    return [{'direction': d, 'text': t} for d, t in reasons[:4]]


def rank_in_portfolio(prob):
    """1 + number of portfolio clients with a strictly higher score."""
    return len(PORTFOLIO_ASC) - bisect.bisect_right(PORTFOLIO_ASC, prob) + 1


def decide(prob):
    bands = CARD['risk_bands']
    if prob >= bands['high']:
        return ('high', 'High risk',
                'Call before the due date and offer an instalment plan. A score this high '
                'puts the client among the 10% riskiest, the share the team can call each month.')
    if prob >= bands['medium']:
        return ('medium', 'Elevated risk',
                'Riskier than the average client but below the call-list cut-off. '
                'An automatic SMS or app reminder before the due date costs almost nothing.')
    return ('low', 'Low risk', 'Nothing to do. Most clients with this history pay on time.')


@app.get('/')
def index():
    val = CARD['validation'][CARD['model_name']]
    stats = {
        'model': CARD['model_name'],
        'auc': CARD['test']['ROC-AUC'],
        'precision_at_k': CARD['test'].get(f'Precision at {CARD["capacity"]}'),
        'capacity': CARD['capacity'],
        'base_rate': CARD['base_default_rate'],
        'val_ap': val['Average Precision'],
        'high_cut': CARD['risk_bands']['high'],
    }
    return render_template('index.html', stats=stats, months=MONTHS)


@app.post('/api/predict')
def predict():
    df, error = read_payload(request.get_json(silent=True) or {})
    if error:
        return jsonify({'error': error}), 400
    prob = float(MODEL.predict_proba(df)[0, 1])
    band, label, action = decide(prob)
    return jsonify({'probability': prob, 'band': band, 'label': label,
                    'action': action, 'reasons': explain(df),
                    'base_rate': CARD['base_default_rate'],
                    'rank': rank_in_portfolio(prob), 'portfolio_size': len(PORTFOLIO),
                    'capacity': CARD['capacity']})


@app.get('/api/portfolio')
def portfolio():
    return jsonify({'scores': PORTFOLIO, 'capacity': CARD['capacity'],
                    'base_rate': CARD['base_default_rate']})


@app.get('/api/model')
def model_info():
    return jsonify({k: CARD[k] for k in ['model_name', 'dataset', 'input_columns',
                                         'excluded_columns', 'test', 'risk_bands']})


if __name__ == '__main__':
    app.run(debug=True)
