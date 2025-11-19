# Requires: pip install yfinance pandas
import yfinance as yf
import pandas as pd

# ---------- Params ----------
ticker = "MARUTI.NS"
# Interpret "Q2 2025" as quarter ending 2025-06-30 (change if needed)
cutoff_quarter_end = pd.to_datetime("2025-06-30")

# ---------- Fetch quarterly income statement ----------
t = yf.Ticker(ticker)
qf = t.quarterly_financials  # rows = line items, cols = quarter end timestamps

# ---------- Helper to find rows robustly ----------
def find_row(df, possible_names):
    idx_lower = [s.lower() for s in df.index]
    # exact-match first
    for name in possible_names:
        if name.lower() in idx_lower:
            return df.loc[df.index[idx_lower.index(name.lower())]]
    # contains-match fallback
    for i in df.index:
        for name in possible_names:
            if name.lower() in i.lower():
                return df.loc[i]
    raise KeyError(f"None of {possible_names} found. Available rows: {list(df.index)}")

# Common candidate labels (tweak if needed)
revenue_row = find_row(qf, ["Total Revenue", "Total revenue", "Revenue", "Net Sales", "TotalNetRevenue"])
net_income_row = find_row(qf, ["Net Income", "Net income", "Net Profit", "NetIncome"])
oper_income_row = find_row(qf, ["Operating Income", "Operating income", "Profit from operations", "EBIT", "Operating Profit"])

# ---------- Build tidy DataFrame ----------
df_q = pd.DataFrame({
    "Revenue": revenue_row,
    "Net Income": net_income_row,
    "Operating Income": oper_income_row
}).T

# transform so quarters are the index
df_quarters = df_q.T.copy()
df_quarters.index = pd.to_datetime(df_quarters.index)
df_quarters = df_quarters.sort_index()

# ensure numeric
df_quarters = df_quarters.apply(pd.to_numeric, errors="coerce")

# ---------- Compute Operating Margin from raw INR values (no scaling) ----------
# Operating Margin (%) = Operating Income / Revenue * 100
df_quarters["Operating_Margin"] = (df_quarters["Operating Income"] / df_quarters["Revenue"]) * 100

# ---------- Convert monetary columns to ₹ Crores (1 crore = 1e7 rupees) ----------
monetary_cols = ["Revenue", "Net Income", "Operating Income"]
df_crores = df_quarters.copy()
df_crores[monetary_cols] = df_crores[monetary_cols] / 1e7   # now in ₹ Crores

# ---------- Select last 6 quarters up to cutoff ----------
sel = df_crores[df_crores.index <= cutoff_quarter_end].tail(6)

# format index as 'Q# YYYY'
sel.index = pd.PeriodIndex(sel.index, freq='Q').strftime('Q%q %Y')

# show only the requested columns
result = sel[["Revenue", "Net Income", "Operating_Margin"]]

print(result)
