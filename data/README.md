# Data: Default of Credit Card Clients

- **Source:** UCI Machine Learning Repository, dataset 350
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Citation:** Yeh, I.-C. & Lien, C.-h. (2009). The comparisons of data mining techniques for the
  predictive accuracy of probability of default of credit card clients. *Expert Systems with
  Applications*, 36(2), 2473–2480.
- **Licence:** CC BY 4.0. The file `default_of_credit_card_clients.xls` is the unchanged original.
- **Content:** 30,000 clients of a Taiwanese bank, April–September 2005, 25 columns, no missing values.
  The first row of the file is a second header, so it is read with `pd.read_excel(path, header=1)`.

## Columns

| Column | Meaning |
|---|---|
| `ID` | Client identifier (not used as an input) |
| `LIMIT_BAL` | Credit limit, NT$ (individual + family) |
| `SEX` | 1 = male, 2 = female (**not used**: protected attribute) |
| `EDUCATION` | 1 = graduate school, 2 = university, 3 = high school, 4 = other; 0, 5, 6 are undocumented and grouped into 4 |
| `MARRIAGE` | 1 = married, 2 = single, 3 = other; 0 is undocumented and grouped into 3 (**not used**) |
| `AGE` | Age in years (**not used**: protected attribute) |
| `PAY_0`, `PAY_2` … `PAY_6` | Repayment status in September, August … April. -2 = no balance, -1 = paid in full, 0 = minimum paid (revolving credit), 1–8 = months late. The codes -2 and 0 are not in the official description; their meaning comes from the dataset's discussion on Kaggle and other public analyses |
| `BILL_AMT1` … `BILL_AMT6` | Bill statement amount, NT$, September … April |
| `PAY_AMT1` … `PAY_AMT6` | Amount paid, NT$, September … April |
| `default payment next month` | **Target**: 1 = the client missed the October 2005 payment (22.1% of clients) |

Note: there is no column named `PAY_1`; the original file jumps from `PAY_0` to `PAY_2`.
