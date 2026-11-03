# %% [markdown]
# # Credit Card Default Prediction — AI for Business (FI510E, Midterm 1)
#
# **Business question.** A credit-card issuer wants to know, one month ahead,
# which clients are likely to miss their next payment, so that a limited
# collections / customer-care team can contact them early (payment reminder,
# instalment plan, limit review) before the debt turns into a loss.
#
# **Dataset.** Default of Credit Card Clients, UCI Machine Learning Repository
# (Yeh & Lien, 2009) — 30,000 clients of a Taiwanese bank, April–September 2005.
# https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
#
# **Positive class = default next month** (the event the business wants to detect).
#
# | Outcome | Meaning |
# |---|---|
# | TP | Future defaulter correctly flagged for early contact |
# | FP | Good client contacted unnecessarily (cost + annoyance) |
# | FN | Future defaulter missed (credit loss) |
# | TN | Good client correctly left alone |
#
# Structure follows the Session 6 laboratory: Task 1 inspect, Task 2 features &
# split, Task 3 five models at threshold 0.50, Task 4 ranking under limited
# capacity, Task 5 threshold trade-off, Task 6 our innovations, Task 7 one-time test.
#
# Run from the repository root:  `python src/train.py`

# %%
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
IN_NOTEBOOK = 'ipykernel' in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use('Agg')          # no window when run as a script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:                      # running inside a notebook
    ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from src.features import (ENGINEERED_COLS, RAW_CATEGORY_COLS, RAW_NUMBER_COLS,
                          SENSITIVE_COLS, add_features, clean)

DATA_FILE = ROOT / 'data' / 'default_of_credit_card_clients.xls'
FIG_DIR = ROOT / 'results' / 'figures'
TABLE_DIR = ROOT / 'results' / 'tables'
MODEL_DIR = ROOT / 'models'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CAPACITY_SHARE = 0.10        # the team can contact 10% of the clients each month
# Business assumptions (to be validated with the bank's finance team)
CONTACT_COST = 500           # NT$ per contact (agent time, SMS, letter)
FALSE_ALARM_COST = 1500      # NT$ extra cost when a GOOD client is contacted
                             # (annoyance, churn risk x lifetime value)
RECOVERY_RATE = 0.10         # share of the outstanding bill saved when a future
                             # defaulter is contacted early — must be confirmed
                             # by an A/B test (see Task 6c)
TARGET = 'default'

plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})
COLORS = {'Logistic Regression': '#7A8B99', 'Decision Tree': '#C9A227',
          'Random Forest': '#2E7D5B', 'Support Vector Machine': '#8E5572',
          'Extreme Gradient Boosting': '#1F4E79'}


def save_figure(filename):
    """Save to results/figures and display it when running in a notebook."""
    plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.show() if IN_NOTEBOOK else plt.close()


def header(text):
    print('\n' + '=' * 78 + f'\n{text}\n' + '=' * 78)


# %% [markdown]
# ## Task 1 — Inspect the data and default risk patterns

# %%
header('TASK 1 — Inspect the data')
clients = pd.read_excel(DATA_FILE, header=1)
clients = clients.rename(columns={'default payment next month': TARGET})

print('Shape:', clients.shape)
print('\nTarget distribution:')
print(clients[TARGET].value_counts().rename({0: 'paid (0)', 1: 'default (1)'}))
base_rate = clients[TARGET].mean()
print(f'Base default rate: {base_rate:.1%}')
print(f'-> A model that predicts "no default" for everybody already has '
      f'{1 - base_rate:.1%} accuracy. Accuracy alone is therefore misleading.')

# --- Data-quality checks -----------------------------------------------------
print('\nData-quality checks')
print('  Missing values           :', int(clients.isna().sum().sum()))
dup = clients.drop(columns='ID').duplicated().sum()
print('  Duplicate rows (excl. ID):', int(dup))
print('  EDUCATION codes          :', sorted(clients['EDUCATION'].unique().tolist()),
      '-> 0, 5, 6 are undocumented, merged into 4 = "other"')
print('  MARRIAGE codes           :', sorted(clients['MARRIAGE'].unique().tolist()),
      '-> 0 is undocumented, merged into 3 = "other"')
print('  PAY_0 codes              :', sorted(clients['PAY_0'].unique().tolist()),
      '-> -2 (no use) and 0 (revolving) are not in the official codebook')
neg_bill = (clients[[f'BILL_AMT{i}' for i in range(1, 7)]] < 0).any(axis=1).sum()
print(f'  Clients with a negative bill (credit balance / overpayment): {neg_bill}')
print('  Amounts are in New Taiwan dollars (NT$) and strongly right-skewed.')

# --- Risk patterns ------------------------------------------------------------
eda = clean(clients)
print('\nObserved default rate by latest repayment status (PAY_0):')
pay0 = eda.groupby(eda['PAY_0'].clip(upper=3))[TARGET].agg(['mean', 'size'])
pay0.index = ['-2 no use', '-1 paid in full', '0 revolving', '1 month late',
              '2 months late', '3+ months late']
print(pay0.rename(columns={'mean': 'default_rate', 'size': 'clients'}).round(3))

for col, labels in [('SEX', {1: 'male', 2: 'female'}),
                    ('EDUCATION', {1: 'graduate school', 2: 'university',
                                   3: 'high school', 4: 'other'}),
                    ('MARRIAGE', {1: 'married', 2: 'single', 3: 'other'})]:
    rate = eda.groupby(col)[TARGET].mean().rename(labels)
    print(f'\nDefault rate by {col}:\n{rate.round(3)}')

print('\nInterpretation: clients who were already late last month default far '
      'more often.\nThese are correlations, not causes: a late payment and a '
      'future default can both\nbe driven by an unobserved cause (job loss, '
      'illness). Differences by sex or\nmarital status must not be read as '
      '"being male causes default" — and in credit\ndecisions they raise legal '
      'and ethical issues (see Task 6b).')

# Figure: default rate by PAY_0
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(pay0.index, pay0['mean'] * 100, color='#1F4E79')
ax.axhline(base_rate * 100, color='#B23A48', ls='--', lw=1)
ax.text(5.4, base_rate * 100 + 2, f'average {base_rate:.0%}', color='#B23A48',
        ha='right')
ax.set_ylabel('Default rate next month (%)')
ax.set_title('Default rate by repayment status in September')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
save_figure('01_default_by_pay0.png')


# %% [markdown]
# ## Task 2 — Approved feature list and data split

# %%
header('TASK 2 — Features and split')
PRODUCTION_NUMBER_COLS = [c for c in RAW_NUMBER_COLS if c not in SENSITIVE_COLS]
PRODUCTION_CATEGORY_COLS = [c for c in RAW_CATEGORY_COLS if c not in SENSITIVE_COLS]
INPUT_COLS = PRODUCTION_NUMBER_COLS + PRODUCTION_CATEGORY_COLS

print('Approved raw inputs :', INPUT_COLS)
print('Engineered features :', ENGINEERED_COLS)
print('Excluded            : ID (identifier, no predictive meaning),')
print('                      default (target — known only next month),')
print('                      SEX, MARRIAGE, AGE (protected attributes; Task 6b '
      'shows the\n                      accuracy cost of removing them is negligible).')
print('All inputs describe April–September 2005 and are known BEFORE the October')
print('payment is due, so there is no information leakage from the future.')

# The dataset has no transaction timestamp: every row is one client observed at
# the same date. A chronological split (as in AtlasPay) is therefore impossible;
# we use a stratified random split, which keeps the 22% default rate in each part.
train, temp = train_test_split(clients, test_size=0.30,
                               stratify=clients[TARGET], random_state=SEED)
valid, test = train_test_split(temp, test_size=0.50,
                               stratify=temp[TARGET], random_state=SEED)
for name, part in [('train', train), ('valid', valid), ('test', test)]:
    print(f'{name:>5}: {len(part):>6,} clients, default rate {part[TARGET].mean():.1%}')
print('Validation = model selection; test = touched ONCE at the very end.')

CAPACITY = int(round(CAPACITY_SHARE * len(valid)))
print(f'Operational capacity: {CAPACITY} contacts per {len(valid):,} clients '
      f'({CAPACITY_SHARE:.0%}).')


def make_pipeline(algorithm, number_cols, category_cols, engineered=True):
    """Feature engineering + scaling/encoding fitted on training data only."""
    numbers = number_cols + (ENGINEERED_COLS if engineered else [])
    prepare = ColumnTransformer([
        ('numbers', StandardScaler(), numbers),
        ('categories', OneHotEncoder(handle_unknown='ignore',
                                     sparse_output=False), category_cols),
    ])
    steps = [('prepare', prepare), ('model', algorithm)]
    if engineered:
        steps.insert(0, ('engineer', FunctionTransformer(add_features)))
    else:
        steps.insert(0, ('clean', FunctionTransformer(clean)))
    return Pipeline(steps)


def make_methods():
    """The five classifiers studied in class (fresh objects each call)."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=10,
                                                min_samples_leaf=20, n_jobs=-1,
                                                random_state=SEED),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, gamma='scale',
                                      probability=True, cache_size=1000,
                                      random_state=SEED),
        'Extreme Gradient Boosting': XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric='logloss', random_state=SEED),
    }


# %% [markdown]
# ## Task 3 — Train five models on the same observations (threshold 0.50)

# %%
header('TASK 3 — Five models, validation scores at threshold 0.50')
models, valid_scores = {}, {}
for name, algorithm in make_methods().items():
    start = time.time()
    model = make_pipeline(algorithm, PRODUCTION_NUMBER_COLS, PRODUCTION_CATEGORY_COLS)
    model.fit(train, train[TARGET])
    models[name] = model
    valid_scores[name] = model.predict_proba(valid)[:, 1]
    print(f'  trained {name:<26} in {time.time() - start:5.1f}s')

y_valid = valid[TARGET].values


def threshold_metrics(y, scores, threshold=0.50):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'F1-score': f1_score(y, pred),
            'Specificity': tn / (tn + fp),
            'Alerts': int(pred.sum()), 'TP': int(tp), 'FP': int(fp),
            'FN': int(fn), 'TN': int(tn),
            'Average Precision': average_precision_score(y, scores),
            'ROC-AUC': roc_auc_score(y, scores)}


val_table = pd.DataFrame({n: threshold_metrics(y_valid, s)
                          for n, s in valid_scores.items()}).T
print(val_table[['Accuracy', 'Precision', 'Recall', 'F1-score', 'Alerts',
                 'Average Precision', 'ROC-AUC']].round(3))
val_table.to_csv(TABLE_DIR / 'validation_threshold_050.csv')

print('\nBusiness interpretation')
print(f'  Fewest missed defaulters (highest recall): {val_table["Recall"].idxmax()} '
      f'(FN = {int(val_table.loc[val_table["Recall"].idxmax(), "FN"])})')
print(f'  Fewest false alerts                       : {val_table["FP"].astype(int).idxmin()} '
      f'(FP = {int(val_table["FP"].min())})')
print(f'  Highest accuracy                          : {val_table["Accuracy"].idxmax()} '
      f'({val_table["Accuracy"].max():.1%}) — only '
      f'{val_table["Accuracy"].max() - (1 - y_valid.mean()):.1%} above "predict nobody".')
print('  At 0.50 every model still misses most defaulters (recall < 50%):')
print('  accuracy is not a suitable selection criterion for default detection.')

# ROC and precision-recall curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, s in valid_scores.items():
    fpr, tpr, _ = roc_curve(y_valid, s)
    prec, rec, _ = precision_recall_curve(y_valid, s)
    axes[0].plot(fpr, tpr, color=COLORS[name], lw=1.6,
                 label=f'{name} (AUC {roc_auc_score(y_valid, s):.3f})')
    axes[1].plot(rec, prec, color=COLORS[name], lw=1.6,
                 label=f'{name} (AP {average_precision_score(y_valid, s):.3f})')
axes[0].plot([0, 1], [0, 1], color='grey', ls=':', lw=1)
axes[1].axhline(y_valid.mean(), color='grey', ls=':', lw=1)
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve')
axes[1].set(xlabel='Recall', ylabel='Precision', title='Precision–recall curve')
for ax in axes:
    ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
save_figure('02_roc_pr_curves.png')


# %% [markdown]
# ## Task 4 — Rank clients under a limited contact capacity

# %%
header(f'TASK 4 — Top {CAPACITY} highest-risk validation clients per model')


def top_k(y, scores, k):
    selected = np.argsort(-scores, kind='stable')[:k]
    found = int(y[selected].sum())
    return {'Real defaulters found': found, 'False alerts': k - found,
            f'Precision at {k}': found / k, f'Recall at {k}': found / y.sum()}


topk_table = pd.DataFrame({n: top_k(y_valid, s, CAPACITY)
                           for n, s in valid_scores.items()}).T
topk_table['Average Precision'] = val_table['Average Precision']
print(topk_table.round(3))
topk_table.to_csv(TABLE_DIR / 'validation_top_k.csv')

# Selection rule: highest Average Precision (quality of the whole ranking),
# Precision@K as tie-breaker. Both are suitable for a 22% minority class.
# Near-tie rule: XGBoost's validation AP moved by ~0.004 between two team laptops
# with the same library versions (0.550 vs 0.554), while Random Forest stayed at
# 0.552. A gap below TIE_MARGIN is therefore noise, not a real difference. Among
# near-tied models we prefer one whose result does not depend on the computer,
# so that the contact list does not change when the model is retrained elsewhere.
TIE_MARGIN = 0.005
MACHINE_DEPENDENT = {'Extreme Gradient Boosting'}
ranking = topk_table.sort_values(['Average Precision', f'Precision at {CAPACITY}'],
                                 ascending=False)
near_best = [m for m in ranking.index
             if ranking.loc[m, 'Average Precision'] >= ranking['Average Precision'].iloc[0] - TIE_MARGIN]
stable = [m for m in near_best if m not in MACHINE_DEPENDENT]
BEST = stable[0] if stable else ranking.index[0]
print(f'\nModels within {TIE_MARGIN} AP of the best (near-tie): {near_best}')
print(f'Candidate model (highest validation AP, stable across computers): {BEST}')
print(f'  finds {int(topk_table.loc[BEST, "Real defaulters found"])} defaulters in '
      f'{CAPACITY} contacts vs ~{int(CAPACITY * y_valid.mean())} with random calls '
      f'(x{topk_table.loc[BEST, f"Precision at {CAPACITY}"] / y_valid.mean():.1f} lift).')


# %% [markdown]
# ## Task 5 — Threshold trade-off vs. "top K" rule

# %%
header('TASK 5 — Threshold policy vs. exact top-K policy')
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
threshold_rows = []
for name in dict.fromkeys(['Support Vector Machine', 'Extreme Gradient Boosting', BEST]):
    print(f'\n{name}')
    for t in thresholds:
        alerts = int((valid_scores[name] >= t).sum())
        ok = 'within capacity' if alerts <= CAPACITY else ''
        print(f'  threshold {t:.2f}: {alerts:>5} alerts {ok}')
        threshold_rows.append({'model': name, 'threshold': t, 'alerts': alerts})
pd.DataFrame(threshold_rows).to_csv(TABLE_DIR / 'threshold_alerts.csv', index=False)
print(f'\nA fixed threshold gives a VARIABLE number of alerts that moves with the')
print(f'portfolio (a recession pushes every score up). With a strict limit of')
print(f'{CAPACITY} contacts, "contact the {CAPACITY} highest scores" is easier to run:')
print('the team is never overloaded and never idle.')


# %% [markdown]
# ## Task 6 — Our innovations beyond the standard project
# **6a.** Feature engineering ablation · **6b.** Fairness experiment ·
# **6c.** Cost–benefit and capacity optimisation

# %%
header('TASK 6a — Do our engineered features help? (validation Average Precision)')
FAST = ['Logistic Regression', 'Decision Tree', 'Random Forest',
        'Extreme Gradient Boosting']          # SVM skipped: ~2 min per fit
ablation = []
for name in FAST:
    raw_model = make_pipeline(make_methods()[name], PRODUCTION_NUMBER_COLS,
                              PRODUCTION_CATEGORY_COLS, engineered=False)
    raw_model.fit(train, train[TARGET])
    raw_s = raw_model.predict_proba(valid)[:, 1]
    ablation.append({'model': name,
                     'AP raw features': average_precision_score(y_valid, raw_s),
                     'AP + engineered': val_table.loc[name, 'Average Precision'],
                     'AUC raw features': roc_auc_score(y_valid, raw_s),
                     'AUC + engineered': val_table.loc[name, 'ROC-AUC']})
ablation = pd.DataFrame(ablation).set_index('model')
ablation['AUC gain'] = ablation['AUC + engineered'] - ablation['AUC raw features']
print(ablation.round(4))
ablation.to_csv(TABLE_DIR / 'innovation_feature_engineering.csv')
print('Ratios such as credit utilisation and repayment ratio help most the simple,')
print('interpretable Logistic Regression; tree ensembles find them partly by themselves.')

# %%
header('TASK 6b — Fairness: what does it cost to drop SEX, MARRIAGE, AGE?')
fair_rows = []
for name in FAST:
    full = make_pipeline(make_methods()[name], RAW_NUMBER_COLS, RAW_CATEGORY_COLS)
    full.fit(train, train[TARGET])
    full_s = full.predict_proba(valid)[:, 1]
    fair_rows.append({'model': name,
                      'AP with sensitive': average_precision_score(y_valid, full_s),
                      'AP without (ours)': val_table.loc[name, 'Average Precision'],
                      'AUC with sensitive': roc_auc_score(y_valid, full_s),
                      'AUC without (ours)': val_table.loc[name, 'ROC-AUC']})
fairness = pd.DataFrame(fair_rows).set_index('model')
fairness['AUC cost'] = fairness['AUC with sensitive'] - fairness['AUC without (ours)']
print(fairness.round(4))
fairness.to_csv(TABLE_DIR / 'innovation_fairness.csv')

# Who ends up on the contact list? Compare flag shares by sex.
sel = np.argsort(-valid_scores[BEST], kind='stable')[:CAPACITY]
flagged = valid.iloc[sel]
group_check = pd.DataFrame({
    'share of portfolio': valid['SEX'].map({1: 'male', 2: 'female'}).value_counts(normalize=True),
    'share of defaulters': valid.loc[valid[TARGET] == 1, 'SEX'].map({1: 'male', 2: 'female'}).value_counts(normalize=True),
    'share of contact list': flagged['SEX'].map({1: 'male', 2: 'female'}).value_counts(normalize=True),
    'precision on list': flagged.groupby(flagged['SEX'].map({1: 'male', 2: 'female'}))[TARGET].mean(),
})
print('\nGroup check of the production model (sex is NOT an input):')
print(group_check.round(3))
group_check.to_csv(TABLE_DIR / 'innovation_group_check.csv')
print('Removing protected attributes costs almost nothing in accuracy, reduces')
print('legal risk, and the contact list stays close to the defaulters\' composition.')

# %%
header('TASK 6c — Cost–benefit: how many clients should we contact?')
exposure_valid = valid['BILL_AMT1'].clip(lower=0).values
print('Cost of each type of error / decision (assumptions):')
print(f'  every contact (TP or FP)   : {CONTACT_COST} NT$')
print(f'  false positive (good client): +{FALSE_ALARM_COST} NT$ goodwill / churn risk')
print(f'  true positive (defaulter)   : saves {RECOVERY_RATE:.0%} of the outstanding bill')
print(f'  false negative (missed)     : the avoidable {RECOVERY_RATE:.0%} is lost')


def net_value(y, scores, exposure, k):
    """Net value (NT$) of contacting the k highest-risk clients."""
    selected = np.argsort(-scores, kind='stable')[:k]
    saved = RECOVERY_RATE * (exposure[selected] * y[selected]).sum()
    false_alarms = k - y[selected].sum()
    return saved - CONTACT_COST * k - FALSE_ALARM_COST * false_alarms, saved


ks = np.arange(50, len(valid) + 1, 50)
curve = pd.DataFrame({'contacts': ks,
                      'net_value_model': [net_value(y_valid, valid_scores[BEST],
                                                    exposure_valid, k)[0] for k in ks]})
mean_saving_random = RECOVERY_RATE * (exposure_valid * y_valid).mean()
random_unit = (mean_saving_random - CONTACT_COST
               - FALSE_ALARM_COST * (1 - y_valid.mean()))
curve['net_value_random'] = ks * random_unit
best_k = int(curve.loc[curve['net_value_model'].idxmax(), 'contacts'])
v_cap, _ = net_value(y_valid, valid_scores[BEST], exposure_valid, CAPACITY)
v_rand = CAPACITY * random_unit
print(f'Net value at capacity ({CAPACITY} contacts): model {v_cap:,.0f} NT$ vs '
      f'random {v_rand:,.0f} NT$')
print(f'Value-maximising number of contacts: {best_k} '
      f'(net {curve["net_value_model"].max():,.0f} NT$)')
if best_k > CAPACITY:
    print(f'-> The {CAPACITY}-contact limit is binding: contacts up to ~{best_k} are '
          f'still profitable,\n   a business case for a larger team or cheaper '
          f'digital reminders (SMS/app).')
else:
    print(f'-> Contacting more than ~{best_k} clients destroys value: false alarms '
          f'cost more than they save.')
curve.to_csv(TABLE_DIR / 'innovation_cost_curve.csv', index=False)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(curve['contacts'], curve['net_value_model'] / 1e6, color='#1F4E79',
        lw=2, label=f'{BEST} ranking')
ax.plot(curve['contacts'], curve['net_value_random'] / 1e6, color='#7A8B99',
        lw=1.5, ls='--', label='Random contacts')
ax.axvline(CAPACITY, color='#B23A48', lw=1)
ax.text(CAPACITY + 40, ax.get_ylim()[0] * 0.85, f'capacity {CAPACITY}', color='#B23A48')
ax.set(xlabel='Number of clients contacted', ylabel='Net value (million NT$)',
       title='Expected net value of early contact (validation set)')
ax.legend(frameon=False)
plt.tight_layout()
save_figure('03_cost_curve.png')

# Which inputs drive the candidate model? (permutation importance on raw inputs)
perm = permutation_importance(models[BEST], valid[INPUT_COLS], y_valid,
                              scoring='average_precision', n_repeats=5,
                              random_state=SEED, n_jobs=1)
importance = (pd.Series(perm.importances_mean, index=INPUT_COLS)
              .sort_values(ascending=False))
print('\nTop drivers of the candidate model (drop in AP when shuffled):')
print(importance.head(8).round(4))
importance.to_csv(TABLE_DIR / 'feature_importance.csv', header=['importance'])

fig, ax = plt.subplots(figsize=(6, 3.8))
top = importance.head(10)[::-1]
ax.barh(top.index, top.values, color='#2E7D5B')
ax.set(xlabel='Drop in Average Precision when shuffled',
       title=f'What drives the {BEST}')
plt.tight_layout()
save_figure('04_feature_importance.png')


# %% [markdown]
# ## Task 7 — Freeze the choice and report the test result once

# %%
header(f'TASK 7 — One-time test of the frozen choice: {BEST}, top-{CAPACITY} policy')
final = models[BEST]
test_scores = final.predict_proba(test)[:, 1]
y_test = test[TARGET].values
test_thr = threshold_metrics(y_test, test_scores)
test_top = top_k(y_test, test_scores, CAPACITY)
v_test, saved_test = net_value(y_test, test_scores,
                               test['BILL_AMT1'].clip(lower=0).values, CAPACITY)
print(f'ROC-AUC {test_thr["ROC-AUC"]:.3f} | Average Precision {test_thr["Average Precision"]:.3f}'
      f' | Accuracy@0.5 {test_thr["Accuracy"]:.3f}')
for k, v in test_top.items():
    print(f'  {k:<24}: {v:.3f}' if isinstance(v, float) else f'  {k:<24}: {v}')
print(f'  Net value of the {CAPACITY} contacts: {v_test:,.0f} NT$ '
      f'(saved {saved_test:,.0f} NT$ minus contact and false-alarm costs)')
print('The test result is reported, not used to change the model: picking a new')
print('winner after seeing the test would make this estimate optimistic again.')

# Review list: client IDs + risk score, WITHOUT the true outcome
order = np.argsort(-test_scores, kind='stable')[:CAPACITY]
review = pd.DataFrame({'rank': np.arange(1, CAPACITY + 1),
                       'client_id': test['ID'].values[order],
                       'risk_score': test_scores[order].round(4)})
review.to_csv(TABLE_DIR / f'Credit_Review_Top{CAPACITY}.csv', index=False)
print(f'Exported results/tables/Credit_Review_Top{CAPACITY}.csv ({len(review)} rows, no target column)')


# %% [markdown]
# ## Save the production model and the model card used by the website

# %%
header('Saving model for the web application')
# Risk bands for the website, derived from VALIDATION scores (not test)
high_cut = float(np.sort(valid_scores[BEST])[::-1][CAPACITY - 1])
feat_train = add_features(train)
model_card = {
    'model_name': BEST,
    'dataset': 'UCI Default of Credit Card Clients (Taiwan, 2005), id 350',
    'input_columns': INPUT_COLS,
    'excluded_columns': ['ID', TARGET] + SENSITIVE_COLS,
    'split': {'train': len(train), 'valid': len(valid), 'test': len(test)},
    'base_default_rate': float(base_rate),
    'capacity': CAPACITY,
    'risk_bands': {'high': high_cut, 'medium': float(base_rate)},
    'assumptions': {'contact_cost_ntd': CONTACT_COST,
                    'false_alarm_cost_ntd': FALSE_ALARM_COST,
                    'recovery_rate': RECOVERY_RATE},
    'validation': {n: {k: float(v) for k, v in row.items()}
                   for n, row in val_table.iterrows()},
    'validation_top_k': {n: {k: float(v) for k, v in row.items()}
                         for n, row in topk_table.iterrows()},
    'test': {**{k: float(v) for k, v in test_thr.items()},
             **{k: float(v) for k, v in test_top.items()},
             'net_value_ntd': float(v_test), 'saved_ntd': float(saved_test)},
    'cost_curve_best_k': best_k,
    'net_value_valid_model': float(v_cap), 'net_value_valid_random': float(v_rand),
    'feature_importance': {k: float(v) for k, v in importance.items()},
    'portfolio_medians': {c: float(feat_train[c].median()) for c in
                          ['UTIL_LAST', 'PAY_RATIO', 'MONTHS_LATE', 'LIMIT_BAL']},
    'innovation_feature_engineering': ablation.round(4).to_dict(orient='index'),
    'innovation_fairness': fairness.round(4).to_dict(orient='index'),
    'trained_at': time.strftime('%Y-%m-%d %H:%M'),
}
joblib.dump(final, MODEL_DIR / 'credit_default_model.joblib', compress=3)
with open(MODEL_DIR / 'model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)
# The website places a new client in this ranked portfolio (validation scores,
# riskiest first): rank <= CAPACITY means the client would be on the call list.
portfolio = np.sort(valid_scores[BEST])[::-1].round(4).tolist()
with open(MODEL_DIR / 'portfolio_scores.json', 'w') as f:
    json.dump({'model_name': BEST, 'capacity': CAPACITY, 'scores': portfolio}, f)
print('Saved models/credit_default_model.joblib, models/model_card.json '
      'and models/portfolio_scores.json')
print('Start the website with:  python app/app.py')
