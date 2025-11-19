# Requires: pip install yfinance pandas matplotlib seaborn
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- params ----------
ticker = "MARUTI.NS"
expected_new_q_end = pd.to_datetime("2025-09-30")   # quarter end for Q2 FY26

# ---------- fetch quarterly_financials ----------
t = yf.Ticker(ticker)
qf = t.quarterly_financials

print("Available quarter-ends from yfinance:", list(qf.columns))
print("Number of quarters returned by yfinance:", len(qf.columns))

# ---------- helper to find rows ----------
def find_row(df, possible_names):
    idx_lower = [s.lower() for s in df.index]
    for name in possible_names:
        if name.lower() in idx_lower:
            return df.loc[df.index[idx_lower.index(name.lower())]]
    for i in df.index:
        for name in possible_names:
            if name.lower() in i.lower():
                return df.loc[i]
    raise KeyError(f"None of {possible_names} found. Available rows: {list(df.index)}")

revenue_row = find_row(qf, ["Total Revenue", "Total revenue", "Revenue", "Net Sales"])
net_income_row = find_row(qf, ["Net Income", "Net income", "Net Profit"])
oper_income_row = find_row(qf, ["Operating Income", "Operating income", "Profit from operations", "EBIT", "Op. EBIT"])

# ---------- build quarter-indexed raw-INR DataFrame ----------
df_q = pd.DataFrame({
    "Revenue": revenue_row,
    "Net Income": net_income_row,
    "Operating Income": oper_income_row
}).T

df_quarters = df_q.T.copy()
df_quarters.index = pd.to_datetime(df_quarters.index)
df_quarters = df_quarters.sort_index()
df_quarters = df_quarters.apply(pd.to_numeric, errors="coerce")

# ---------- compute Operating Margin (%) from raw-INR ----------
df_quarters["Operating_Margin"] = (df_quarters["Operating Income"] / df_quarters["Revenue"]) * 100

# ---------- convert monetary columns to ₹ Crores (1 crore = 1e7 rupees) ----------
monetary_cols = ["Revenue", "Net Income", "Operating Income"]
df_crores = df_quarters.copy()
df_crores[monetary_cols] = df_crores[monetary_cols] / 1e7   # now in ₹ Crores

# ---------- If expected quarter (2025-09-30) is missing, append official IR numbers ----------
# Official Q2 FY26 (from Maruti IR / investor presentation, 31 Oct 2025):
# Net Sales = 401,359 million INR -> 401,359 / 10 = 40,135.9 crores
# Op. EBIT = 33,949 million INR -> 3,394.9 crores
# PAT = 32,931 million INR -> 3,293.1 crores
if expected_new_q_end not in df_crores.index:
    print("Expected Q2 FY26 (2025-09-30) missing from yfinance; appending official IR numbers (Q2 FY26).")
    revenue_crores = 401359.0 / 10.0    # 401,359 million -> 40,135.9 crores
    oper_income_crores = 33949.0 / 10.0 # 33,949 million -> 3,394.9 crores
    net_income_crores = 32931.0 / 10.0  # 32,931 million -> 3,293.1 crores
    op_margin_pct = (oper_income_crores / revenue_crores) * 100.0

    # create a one-row DataFrame with the same columns
    new_row = pd.DataFrame({
        "Revenue": [revenue_crores],
        "Net Income": [net_income_crores],
        "Operating Income": [oper_income_crores],
        "Operating_Margin": [op_margin_pct]
    }, index=[expected_new_q_end])

    # append and sort
    df_crores = pd.concat([df_crores, new_row]).sort_index()

# ---------- select last 6 quarters up to expected_new_q_end (inclusive) ----------
sel = df_crores[df_crores.index <= expected_new_q_end].tail(6)
if sel.shape[0] < 1:
    raise ValueError("No quarterly data found up to expected_new_q_end. Inspect `qf` for available columns/labels.")

# ---------- fiscal quarter label helper ----------
def fiscal_quarter_label(ts):
    # ts is a Timestamp representing quarter end
    m = ts.month
    y = ts.year
    if m <= 3:            # Jan-Mar => Q4 of FY = year (e.g., 2025-03-31 -> Q4 FY25)
        q = 4
        fy = y
    elif m <= 6:          # Apr-Jun => Q1 of next FY (e.g., 2025-06-30 -> Q1 FY26)
        q = 1
        fy = y + 1
    elif m <= 9:          # Jul-Sep => Q2 of next FY
        q = 2
        fy = y + 1
    else:                 # Oct-Dec => Q3 of next FY
        q = 3
        fy = y + 1
    return f"Q{q} FY{str(fy)[-2:]}"

# ---------- prepare df_plot and use fiscal labels ----------
df_plot = sel.copy()
df_plot['Quarter_End'] = df_plot.index
df_plot = df_plot.reset_index(drop=True)
df_plot['Quarter_Label'] = df_plot['Quarter_End'].apply(fiscal_quarter_label)

# rename to match your plotting code
df_plot = df_plot.rename(columns={"Net Income": "Net_Income"})

# ---------- growth rates ----------
df_plot['Revenue_Growth'] = df_plot['Revenue'].pct_change() * 100
df_plot['Profit_Growth'] = df_plot['Net_Income'].pct_change() * 100

# ---------- (plotting code same as before) ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0,0].plot(df_plot['Quarter_End'], df_plot['Revenue'], marker='o', linewidth=2)
axes[0,0].set_title('Revenue Trend')
axes[0,0].set_ylabel('Revenue (₹ Crores)')
axes[0,0].set_xticks(df_plot['Quarter_End'])
axes[0,0].set_xticklabels(df_plot['Quarter_Label'], rotation=45)
axes[0,0].grid(True, alpha=0.3)

axes[0,1].bar(df_plot['Quarter_End'], df_plot['Revenue_Growth'], alpha=0.7)
axes[0,1].set_title('Revenue Growth Rate (%)')
axes[0,1].set_ylabel('Growth Rate (%)')
axes[0,1].set_xticks(df_plot['Quarter_End'])
axes[0,1].set_xticklabels(df_plot['Quarter_Label'], rotation=45)

axes[1,0].plot(df_plot['Quarter_End'], df_plot['Operating_Margin'], marker='s', linewidth=2)
axes[1,0].set_title('Operating Margin Trend')
axes[1,0].set_ylabel('Margin (%)')
axes[1,0].set_xticks(df_plot['Quarter_End'])
axes[1,0].set_xticklabels(df_plot['Quarter_Label'], rotation=45)

correlation_data = df_plot[['Revenue', 'Net_Income', 'Operating_Margin']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Financial Metrics Correlation')

plt.tight_layout()
plt.show()

# ---------- key metrics ----------
avg_rev_growth = df_plot['Revenue_Growth'].mean(skipna=True)
n_quarters = df_plot.shape[0]
years = n_quarters / 4.0
if n_quarters >= 2 and df_plot['Revenue'].iloc[0] > 0:
    revenue_cagr = ((df_plot['Revenue'].iloc[-1] / df_plot['Revenue'].iloc[0]) ** (1/years) - 1) * 100
else:
    revenue_cagr = float('nan')

print(f"Average Revenue Growth (last {n_quarters} quarters): {avg_rev_growth:.1f}%")
print(f"Revenue CAGR over {years:.2f} years: {revenue_cagr:.1f}%")

print("\nQuarterly table (Revenue & Net Income in ₹ Crores):\n")
display_cols = ["Quarter_Label", "Revenue", "Net_Income", "Operating_Margin", "Revenue_Growth", "Profit_Growth"]
print(df_plot[display_cols].to_string(index=False, float_format='{:,.2f}'.format))
