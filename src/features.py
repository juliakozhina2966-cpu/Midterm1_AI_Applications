"""
Shared data-preparation logic.

Used by BOTH the training script (src/train.py) and the web app (app/app.py),
so that a client typed into the website is transformed exactly like the
training data. Keeping it in one place prevents "training/serving skew".
"""
import numpy as np
import pandas as pd

PAY_STATUS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL_AMT = [f'BILL_AMT{i}' for i in range(1, 7)]
PAY_AMT = [f'PAY_AMT{i}' for i in range(1, 7)]

# Raw inputs a credit analyst (or the website) must supply
RAW_NUMBER_COLS = ['LIMIT_BAL', 'AGE'] + PAY_STATUS + BILL_AMT + PAY_AMT
RAW_CATEGORY_COLS = ['SEX', 'EDUCATION', 'MARRIAGE']

# Features created by our own feature engineering (innovation #1)
ENGINEERED_COLS = ['UTIL_LAST', 'UTIL_AVG', 'PAY_RATIO', 'MONTHS_LATE',
                   'MAX_DELAY', 'BILL_TREND', 'ZERO_PAY_MONTHS']

# Protected attributes: legally sensitive in credit decisions
SENSITIVE_COLS = ['SEX', 'MARRIAGE', 'AGE']


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the undocumented codes found during data analysis."""
    df = df.copy()
    # EDUCATION: 0, 5, 6 are not described by the data provider -> 4 'other'
    if 'EDUCATION' in df:
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    # MARRIAGE: 0 is not described -> 3 'other'
    if 'MARRIAGE' in df:
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-meaningful ratios from the raw monthly history."""
    df = clean(df)
    limit = df['LIMIT_BAL'].clip(lower=1)
    bills = df[BILL_AMT]
    pays = df[PAY_AMT]

    # How much of the credit line is used (last month, 6-month average)
    df['UTIL_LAST'] = (df['BILL_AMT1'] / limit).clip(-1, 3)
    df['UTIL_AVG'] = (bills.mean(axis=1) / limit).clip(-1, 3)

    # Share of the previous bills that was actually repaid
    # (PAY_AMT1 in September repays the August bill BILL_AMT2, etc.)
    paid = df[PAY_AMT[:5]].sum(axis=1)
    owed = df[BILL_AMT[1:]].clip(lower=0).sum(axis=1)
    ratio = paid / owed.where(owed > 0, 1)
    df['PAY_RATIO'] = ratio.where(owed > 0, 1.0).clip(0, 2)

    # Repayment behaviour
    status = df[PAY_STATUS]
    df['MONTHS_LATE'] = (status > 0).sum(axis=1)
    df['MAX_DELAY'] = status.max(axis=1).clip(lower=0)
    df['BILL_TREND'] = ((df['BILL_AMT1'] - df['BILL_AMT6']) / limit).clip(-3, 3)
    df['ZERO_PAY_MONTHS'] = ((pays.values == 0) & (bills.values > 0)).sum(axis=1)
    return df
