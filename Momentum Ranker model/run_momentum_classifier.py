#!/usr/bin/env python3
# run_regressor_selection_pipeline_permfix_ranker_fix_cross_ticker_with_save.py
"""
Analyst :- Kush Milkesh Mistry
ML Model name :- Momentum Ranker

This model looks at past price moves and technical indicators to estimate which dates/tickers are most likely to produce positive returns over a short future horizon (default 5 trading days). 
It trains a LightGBM ranker/regressor in a monthly walk-forward way, selects the top candidates each month (top-k) and/or by percentile, and evaluates performance with backtests and permutation feature importance. 
The pipeline also saves a final model and metadata so you can load it later to score new data and produce buy signals.

"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import copy
import json
import joblib
import datetime

# ---------------- USER PARAMETERS ----------------
# Default ticker (kept for compatibility) - cross-ticker runner will override
TICKER = "MARUTI.NS"
START = "2015-01-01"
END = None
HORIZON = 5
min_train_size = 60

k_values = [1,2,3,4,5,7,10]

# CPU-saving/dev settings (increase for final)
n_estimators_monthly = 100
RANDOM_STATE = 42

# Toggle: use Ranker (True) or point regressor (False)
USE_RANKER = True

# Transaction cost sweep (round-trip fraction)
cost_list = [0.0005, 0.001, 0.002]
PRIMARY_COST = 0.001

# Threshold grid for classification probability sweep (kept for classifier runs)
p_grid = np.linspace(0.40, 0.95, 12)

# Bootstrap for p-value
BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 42

# Regressor params and early stopping
REG_PARAMS = {
    'n_estimators': n_estimators_monthly,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'objective': 'huber',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'verbosity': -1
}
REG_EARLY_STOP = 40

# Ranker params (start from reg params but remove 'objective')
RANKER_PARAMS = copy.deepcopy(REG_PARAMS)
RANKER_PARAMS.pop('objective', None)
RANKER_PARAMS['n_estimators'] = n_estimators_monthly

# Classifier params (kept for comparison)
CLF_PARAMS = {
    'n_estimators': n_estimators_monthly,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'n_jobs': 1,
    'verbosity': -1
}

# Permutation importance for regressor (optional)
PERM_TOP_K = int(np.median(k_values))
PERM_N_REPEATS = 3
PERM_RANDOM_SEED = 123
TOP_N = 8   # number of top features to select for SELECTED_TOPN_REG run

# Winsorization / clipping policy for regressor target on training folds:
TARGET_CLIP_MODE = 'quantile'  # 'quantile' or 'abs'
TARGET_CLIP_QUANTILES = (0.01, 0.99)  # if using quantile mode
TARGET_CLIP_ABS = (-0.10, 0.10)       # if using abs mode

# Ranker-specific: how many discrete relevance bins to create (must be >=2)
RANKER_N_BINS = 5

# Cross-ticker list (5 diversified large-caps) - change if you want different tickers
CROSS_TICKERS = ["BHARTIARTL.NS", "WIPRO.NS", "MARUTI.NS", "HDFCBANK.NS", "RELIANCE.NS"]

# If True, in cross-ticker runs we use fewer permutation repeats to speed up.
PERM_REPEATS_FOR_CROSS_TICKER = 1

# ---------------- FEATURE SETS (missing variables added back) ----------------
baseline_feature_cols = [
    'ret_1','ret_3','ret_5',
    'sma10_sma50','ema_diff','rsi14','macd_hist','atr14','bb_percent','obv',
    'vol_10','sma50_slope'
]

engineered_feature_cols = [
    'ret_1','ret_3','ret_5','ret_10','ret_21','ret_63','ret_126',
    'vol_10','vol_21','vol_63',
    'ret_21_volnorm','ret_63_volnorm','ret_21_z','ret_63_z',
    'sma10_sma50','ema_diff','rsi14','macd_hist','atr14','bb_percent','obv','sma50_slope'
]

# ------------------------------------------------------------------------

Path("results").mkdir(parents=True, exist_ok=True)

# ------------------ Helper to prepare data & features for a ticker ------------------
def prepare_df_for_ticker(ticker):
    """
    Downloads OHLCV for ticker and computes the feature-engineering steps.
    Returns the dataframe with features and target columns created exactly as before.
    """
    print(f"\nDownloading data for {ticker}")
    df_local = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)

    if df_local is None or df_local.shape[0] == 0:
        raise SystemExit(f"No data downloaded for {ticker}. Check ticker or internet connection.")

    # Flatten MultiIndex columns
    if isinstance(df_local.columns, pd.MultiIndex):
        df_local.columns = ['_'.join(map(str,c)).strip() for c in df_local.columns]

    # robust OHLCV mapping
    cols_str = [str(c) for c in df_local.columns]
    cols_lower = [c.lower() for c in cols_str]
    def find_col(key_words):
        for kw in key_words:
            for original, lower in zip(cols_str, cols_lower):
                if kw in lower:
                    return original
        return None

    open_col = find_col(['open'])
    high_col = find_col(['high'])
    low_col = find_col(['low'])
    close_col = find_col(['close','adj close','adjclose','adj_close'])
    volume_col = find_col(['volume','vol'])
    mapped = {'Open': open_col, 'High': high_col, 'Low': low_col, 'Close': close_col, 'Volume': volume_col}
    missing = [k for k,v in mapped.items() if v is None]
    if missing:
        raise SystemExit(f"Could not map columns for: {missing}. Available: {cols_str}")

    df_local = df_local[[mapped['Open'], mapped['High'], mapped['Low'], mapped['Close'], mapped['Volume']]].copy()
    df_local.columns = ['Open','High','Low','Close','Volume']

    # ensure numeric
    for col in ['Open','High','Low','Close','Volume']:
        col_data = df_local[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:,0]
        if isinstance(col_data, (np.ndarray, list)):
            col_data = pd.Series(np.squeeze(col_data), index=df_local.index)
        df_local[col] = pd.to_numeric(col_data, errors='coerce')

    # ================= FEATURE ENGINEERING =================
    close = df_local['Close'].astype(float)

    # multi-horizon returns
    df_local['ret_1'] = close.pct_change(1)
    df_local['ret_3'] = close.pct_change(3)
    df_local['ret_5'] = close.pct_change(5)
    df_local['ret_10'] = close.pct_change(10)
    df_local['ret_21'] = close.pct_change(21)
    df_local['ret_63'] = close.pct_change(63)
    df_local['ret_126'] = close.pct_change(126)

    # rolling vol
    df_local['vol_10'] = close.pct_change().rolling(10).std()
    df_local['vol_21'] = close.pct_change().rolling(21).std()
    df_local['vol_63'] = close.pct_change().rolling(63).std()

    eps = 1e-9
    df_local['ret_21_volnorm'] = df_local['ret_21'] / (df_local['vol_21'] + eps)
    df_local['ret_63_volnorm'] = df_local['ret_63'] / (df_local['vol_63'] + eps)

    # z-scored momentum features (252-day rolling)
    df_local['ret_21_z'] = (df_local['ret_21'] - df_local['ret_21'].rolling(252).mean()) / (df_local['ret_21'].rolling(252).std() + eps)
    df_local['ret_63_z'] = (df_local['ret_63'] - df_local['ret_63'].rolling(252).mean()) / (df_local['ret_63'].rolling(252).std() + eps)

    # classic technical indicators
    df_local['sma_10'] = SMAIndicator(close, window=10).sma_indicator()
    df_local['sma_50'] = SMAIndicator(close, window=50).sma_indicator()
    df_local['sma_200'] = SMAIndicator(close, window=200).sma_indicator()
    df_local['sma10_sma50'] = df_local['sma_10'] / df_local['sma_50']

    df_local['ema12'] = EMAIndicator(close, window=12).ema_indicator()
    df_local['ema26'] = EMAIndicator(close, window=26).ema_indicator()
    df_local['ema_diff'] = df_local['ema12'] - df_local['ema26']

    df_local['rsi14'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df_local['macd'] = macd.macd()
    df_local['macd_signal'] = macd.macd_signal()
    df_local['macd_hist'] = macd.macd_diff()

    df_local['atr14'] = AverageTrueRange(df_local['High'], df_local['Low'], close, window=14).average_true_range()
    bb = BollingerBands(close, window=20, window_dev=2)
    df_local['bb_percent'] = bb.bollinger_pband()

    df_local['obv'] = (np.sign(close.diff()) * df_local['Volume']).fillna(0).cumsum()
    df_local['sma50_slope'] = df_local['sma_50'].pct_change(5)

    # targets
    df_local['future_close'] = close.shift(-HORIZON)
    df_local['future_return'] = df_local['future_close'] / close - 1

    return df_local

# ------------------ (Rest of pipeline functions are unchanged, copied verbatim) ------------------
def simple_backtest_from_preds(pred_dates, df_local, HORIZON=5):
    pos = pd.Series(0, index=df_local.index, dtype=int)
    for dt in pred_dates:
        if dt not in df_local.index:
            continue
        i = df_local.index.get_loc(dt)
        exit_i = i + HORIZON
        if exit_i < len(df_local):
            pos.iloc[i:exit_i+1] = 1
    strat_ret = pos.shift(1) * df_local['Close'].pct_change().fillna(0)
    cum = (1 + strat_ret).cumprod()
    return cum, strat_ret, pos

def ann_stats_from_daily_returns(r):
    r = r.dropna()
    if len(r) == 0:
        return (0.0, 0.0, 0.0)
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return (float(ann_ret), float(ann_vol), float(sharpe))

def apply_roundtrip_costs_to_strat_returns(strat_ret_series, predicted_dates, df_local, roundtrip_cost):
    adj = strat_ret_series.copy().astype(float)
    pos = pd.Series(0, index=df_local.index, dtype=int)
    for dt in predicted_dates:
        if dt not in df_local.index:
            continue
        i = df_local.index.get_loc(dt)
        exit_i = i + HORIZON
        if exit_i < len(df_local):
            pos.iloc[i:exit_i+1] = 1
    entry_mask = (pos.diff().fillna(pos) == 1)
    entry_dates = pos.index[entry_mask]
    for d in entry_dates:
        if d in adj.index:
            adj.loc[d] = adj.loc[d] - roundtrip_cost
    return adj, pos

# ------------------ CLASSIFIER (for comparison) ------------------
def _train_and_calibrate_predict_clf(X, y_sign, train_inner_idx, cal_idx, test_idx):
    X_tr = X.loc[train_inner_idx]
    y_tr = y_sign.loc[train_inner_idx]
    X_cal = X.loc[cal_idx]
    y_cal = y_sign.loc[cal_idx]
    X_te = X.loc[test_idx]

    clf = LGBMClassifier(**CLF_PARAMS)
    try:
        clf.fit(X_tr, y_tr, verbose=False)
    except TypeError:
        clf.fit(X_tr, y_tr)
    except Exception:
        clf.fit(X_tr, y_tr)

    probs = None
    if len(cal_idx) >= 20:
        try:
            calibrator = CalibratedClassifierCV(estimator=clf, cv='prefit', method='sigmoid')
        except TypeError:
            try:
                calibrator = CalibratedClassifierCV(base_estimator=clf, cv='prefit', method='sigmoid')
            except Exception:
                calibrator = None
        if calibrator is not None:
            try:
                calibrator.fit(X_cal, y_cal)
                probs = pd.Series(calibrator.predict_proba(X_te)[:,1], index=X_te.index)
            except Exception:
                probs = pd.Series(clf.predict_proba(X_te)[:,1], index=X_te.index)
    if probs is None:
        probs = pd.Series(clf.predict_proba(X_te)[:,1], index=X_te.index)
    return probs

def run_calibrated_class_topk_for_X(X, y_sign, df_local, k, require_calibration=True, verbose=False):
    months = pd.period_range(start=X.index.min(), end=X.index.max(), freq='M')
    predicted_dates = []
    prob_values_all = pd.Series(np.nan, index=X.index, dtype=float)
    processed_test_indices = []

    for m in months:
        month_start = m.to_timestamp(how='start')
        month_end = m.to_timestamp(how='end')
        train_idx_all = X.index[X.index < month_start]
        test_idx = X.index[(X.index >= month_start) & (X.index <= month_end)]
        if len(train_idx_all) < min_train_size or len(test_idx) == 0:
            continue
        n_train = len(train_idx_all)
        cal_split = max(1, int(n_train * 0.2))
        train_inner_idx = train_idx_all[:(n_train - cal_split)]
        cal_idx = train_idx_all[(n_train - cal_split):]

        probs = _train_and_calibrate_predict_clf(X, y_sign, train_inner_idx, cal_idx, test_idx)
        prob_values_all.loc[test_idx] = probs.values
        topk = probs.sort_values(ascending=False).head(k).index.tolist()
        if len(topk) > 0:
            predicted_dates.extend(topk)
        processed_test_indices.extend(list(test_idx))

    predicted_dates = sorted(set(predicted_dates))
    processed_test_indices = sorted(set(processed_test_indices))
    cum_pf, strat_ret, pos = simple_backtest_from_preds(predicted_dates, df_local, HORIZON=HORIZON)
    ann = ann_stats_from_daily_returns(strat_ret)
    coverage = len(predicted_dates)/len(df_local) if len(df_local)>0 else 0.0
    exposure_frac = pos.mean()
    entry_dates = [d for d in predicted_dates if d in df_local.index]
    per_trade_returns = df_local.loc[entry_dates, 'future_return'].dropna() if len(entry_dates)>0 else pd.Series([], dtype=float)

    result = {
        'k': k,
        'n_trades': len(entry_dates),
        'oos_days_covered': len(processed_test_indices),
        'cum_return': float(cum_pf.iloc[-1]) if len(cum_pf)>0 else 1.0,
        'ann_return': ann[0],
        'ann_vol': ann[1],
        'sharpe': ann[2],
        'coverage_frac': coverage,
        'exposure_frac': float(exposure_frac),
        'per_trade_count': int(len(per_trade_returns)),
        'per_trade_mean': float(per_trade_returns.mean()) if len(per_trade_returns)>0 else np.nan,
        'per_trade_winrate': float((per_trade_returns>0).mean()) if len(per_trade_returns)>0 else np.nan
    }
    if verbose:
        print("k=", k, "-> n_trades=", result['n_trades'], "ann_ret=", result['ann_return'], "sharpe=", result['sharpe'])
    return result, cum_pf, strat_ret, predicted_dates, prob_values_all

# ------------------ REGRESSOR / RANKER (walk-forward) ------------------
def _groups_from_index(idx_index):
    if len(idx_index) == 0:
        return []
    periods = pd.Series(idx_index.to_period('M'), index=idx_index)
    grp_counts = periods.groupby(periods).size().sort_index().tolist()
    return grp_counts

def _train_and_predict_regressor(X, y_reg, train_inner_idx, cal_idx, test_idx=None, X_test_df=None, reg_params=REG_PARAMS, early_stopping_rounds=REG_EARLY_STOP):
    X_tr = X.loc[train_inner_idx]
    y_tr = y_reg.loc[train_inner_idx].copy()
    X_cal = X.loc[cal_idx]
    y_cal = y_reg.loc[cal_idx].copy()

    if X_test_df is not None:
        X_te = X_test_df.copy()
    else:
        if test_idx is None:
            raise ValueError("Either test_idx or X_test_df must be provided")
        X_te = X.loc[test_idx]

    if TARGET_CLIP_MODE == 'quantile':
        low_q, high_q = TARGET_CLIP_QUANTILES
        lo = y_tr.quantile(low_q)
        hi = y_tr.quantile(high_q)
        y_tr_clip = y_tr.clip(lo, hi)
    else:
        lo, hi = TARGET_CLIP_ABS
        y_tr_clip = y_tr.clip(lo, hi)

    reg = LGBMRegressor(**reg_params)
    try:
        reg.fit(X_tr, y_tr_clip, eval_set=[(X_cal, y_cal)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    except TypeError:
        try:
            reg.fit(X_tr, y_tr_clip, eval_set=[(X_cal, y_cal)], early_stopping_rounds=early_stopping_rounds)
        except Exception:
            reg.fit(X_tr, y_tr_clip)
    except Exception:
        reg.fit(X_tr, y_tr_clip)

    preds = pd.Series(reg.predict(X_te), index=X_te.index)
    return preds

def _train_and_predict_ranker(X, y_reg, train_inner_idx, cal_idx, test_idx=None, X_test_df=None, rank_params=RANKER_PARAMS):
    X_tr = X.loc[train_inner_idx]
    y_tr = y_reg.loc[train_inner_idx].copy()
    X_cal = X.loc[cal_idx]
    y_cal = y_reg.loc[cal_idx].copy()

    if X_test_df is not None:
        X_te = X_test_df.copy()
    else:
        if test_idx is None:
            raise ValueError("Either test_idx or X_test_df must be provided")
        X_te = X.loc[test_idx]

    # Winsorize/clip on y_tr (same as regressor)
    if TARGET_CLIP_MODE == 'quantile':
        low_q, high_q = TARGET_CLIP_QUANTILES
        lo = y_tr.quantile(low_q)
        hi = y_tr.quantile(high_q)
        y_tr_clip = y_tr.clip(lo, hi)
    else:
        lo, hi = TARGET_CLIP_ABS
        y_tr_clip = y_tr.clip(lo, hi)

    # Convert to integer relevance labels using quantile-based bins on ranked values
    try:
        ranked = y_tr_clip.rank(method='first')
        # If there are fewer rows than bins, qcut will fail; handle that by reducing bins.
        n_bins = min(RANKER_N_BINS, max(2, int(len(ranked) // 2)))
        labels_tr = pd.qcut(ranked, q=n_bins, labels=False, duplicates='drop').astype(int)
    except Exception:
        # fallback: simple integer ranks
        labels_tr = (y_tr_clip.rank(method='first') > y_tr_clip.rank(method='first').median()).astype(int)

    groups_tr = _groups_from_index(X_tr.index)
    # Fit ranker: do not pass early_stopping_rounds to keep compatibility with older LGBM versions
    ranker = LGBMRanker(**rank_params)
    try:
        ranker.fit(X_tr, labels_tr, group=groups_tr)
    except Exception as e:
        try:
            ranker.fit(X_tr, labels_tr)
        except Exception:
            raise

    preds = pd.Series(ranker.predict(X_te), index=X_te.index)
    return preds

def _train_and_predict_model(X, y_reg, train_inner_idx, cal_idx, test_idx=None, X_test_df=None, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP):
    if USE_RANKER:
        return _train_and_predict_ranker(X, y_reg, train_inner_idx, cal_idx, test_idx=test_idx, X_test_df=X_test_df, rank_params=rank_params)
    else:
        return _train_and_predict_regressor(X, y_reg, train_inner_idx, cal_idx, test_idx=test_idx, X_test_df=X_test_df, reg_params=reg_params, early_stopping_rounds=early_stopping_rounds)

def run_regressor_topk_for_X(X, y_reg, df_local, k, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP, verbose=False):
    months = pd.period_range(start=X.index.min(), end=X.index.max(), freq='M')
    predicted_dates = []
    pred_values_all = pd.Series(np.nan, index=X.index, dtype=float)
    processed_test_indices = []

    for m in months:
        month_start = m.to_timestamp(how='start')
        month_end = m.to_timestamp(how='end')
        train_idx_all = X.index[X.index < month_start]
        test_idx = X.index[(X.index >= month_start) & (X.index <= month_end)]
        if len(train_idx_all) < min_train_size or len(test_idx) == 0:
            continue
        n_train = len(train_idx_all)
        cal_split = max(1, int(n_train * 0.2))
        train_inner_idx = train_idx_all[:(n_train - cal_split)]
        cal_idx = train_idx_all[(n_train - cal_split):]

        preds = _train_and_predict_model(X, y_reg, train_inner_idx, cal_idx, test_idx=test_idx, X_test_df=None, reg_params=reg_params, rank_params=rank_params, early_stopping_rounds=early_stopping_rounds)
        pred_values_all.loc[test_idx] = preds.values
        topk = preds.sort_values(ascending=False).head(k).index.tolist()
        if len(topk) > 0:
            predicted_dates.extend(topk)
        processed_test_indices.extend(list(test_idx))

    predicted_dates = sorted(set(predicted_dates))
    processed_test_indices = sorted(set(processed_test_indices))

    cum_pf, strat_ret, pos = simple_backtest_from_preds(predicted_dates, df_local, HORIZON=HORIZON)
    ann = ann_stats_from_daily_returns(strat_ret)
    coverage = len(predicted_dates)/len(df_local) if len(df_local)>0 else 0.0
    exposure_frac = pos.mean()
    entry_dates = [d for d in predicted_dates if d in df_local.index]
    per_trade_returns = df_local.loc[entry_dates, 'future_return'].dropna() if len(entry_dates)>0 else pd.Series([], dtype=float)

    result = {
        'k': k,
        'n_trades': len(entry_dates),
        'oos_days_covered': len(processed_test_indices),
        'cum_return': float(cum_pf.iloc[-1]) if len(cum_pf)>0 else 1.0,
        'ann_return': ann[0],
        'ann_vol': ann[1],
        'sharpe': ann[2],
        'coverage_frac': coverage,
        'exposure_frac': float(exposure_frac),
        'per_trade_count': int(len(per_trade_returns)),
        'per_trade_mean': float(per_trade_returns.mean()) if len(per_trade_returns)>0 else np.nan,
        'per_trade_winrate': float((per_trade_returns>0).mean()) if len(per_trade_returns)>0 else np.nan
    }
    if verbose:
        print("k=", k, "-> n_trades=", result['n_trades'], "ann_ret=", result['ann_return'], "sharpe=", result['sharpe'])
    return result, cum_pf, strat_ret, predicted_dates, pred_values_all

def get_regressor_pred_series_for_X(X, y_reg, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP):
    months = pd.period_range(start=X.index.min(), end=X.index.max(), freq='M')
    pred_values_all = pd.Series(np.nan, index=X.index, dtype=float)
    processed_test_indices = []
    for m in months:
        month_start = m.to_timestamp(how='start')
        month_end = m.to_timestamp(how='end')
        train_idx_all = X.index[X.index < month_start]
        test_idx = X.index[(X.index >= month_start) & (X.index <= month_end)]
        if len(train_idx_all) < min_train_size or len(test_idx) == 0:
            continue
        n_train = len(train_idx_all)
        cal_split = max(1, int(n_train * 0.2))
        train_inner_idx = train_idx_all[:(n_train - cal_split)]
        cal_idx = train_idx_all[(n_train - cal_split):]
        preds = _train_and_predict_model(X, y_reg, train_inner_idx, cal_idx, test_idx=test_idx, X_test_df=None, reg_params=reg_params, rank_params=rank_params, early_stopping_rounds=early_stopping_rounds)
        pred_values_all.loc[test_idx] = preds.values
        processed_test_indices.extend(list(test_idx))
    processed_test_indices = sorted(set(processed_test_indices))
    return pred_values_all, processed_test_indices

# ------------------ Walk-forward permutation importance (regressor/ranker) ------------------
def walkforward_permutation_importance_regressor(df_run, X_run, y_reg_run,
                                                feature_cols,
                                                top_k=PERM_TOP_K,
                                                n_repeats=PERM_N_REPEATS,
                                                random_seed=PERM_RANDOM_SEED,
                                                reg_params=REG_PARAMS,
                                                rank_params=RANKER_PARAMS,
                                                early_stopping_rounds=REG_EARLY_STOP):
    """
    Exactly same logic as original: compute walk-forward permutation importance.
    Exposes n_repeats so callers (cross-ticker runner) can speed it up by lowering repeats.
    """
    rng_master = np.random.RandomState(random_seed)
    months = pd.period_range(start=X_run.index.min(), end=X_run.index.max(), freq='M')
    feat_names = list(feature_cols)
    fold_importances = {f: [] for f in feat_names}
    n_folds = 0

    for m in months:
        month_start = m.to_timestamp(how='start')
        month_end = m.to_timestamp(how='end')
        train_idx_all = X_run.index[X_run.index < month_start]
        test_idx = X_run.index[(X_run.index >= month_start) & (X_run.index <= month_end)]
        if len(train_idx_all) < min_train_size or len(test_idx) == 0:
            continue
        n_train = len(train_idx_all)
        cal_split = max(1, int(n_train * 0.2))
        train_inner_idx = train_idx_all[:(n_train - cal_split)]
        cal_idx = train_idx_all[(n_train - cal_split):]

        preds = _train_and_predict_model(X_run, y_reg_run, train_inner_idx, cal_idx, test_idx=test_idx, X_test_df=None, reg_params=reg_params, rank_params=rank_params, early_stopping_rounds=early_stopping_rounds)
        topk_idx = preds.sort_values(ascending=False).head(top_k).index.tolist()
        if len(topk_idx) == 0:
            continue
        cum_pf_base, strat_ret_base, pos_base = simple_backtest_from_preds(topk_idx, df_run, HORIZON=HORIZON)
        adj_base, _ = apply_roundtrip_costs_to_strat_returns(strat_ret_base.fillna(0), topk_idx, df_run, roundtrip_cost=PRIMARY_COST)
        base_sharpe = ann_stats_from_daily_returns(adj_base)[2]
        if np.isnan(base_sharpe):
            continue

        X_te_orig = X_run.loc[test_idx].copy()
        for f in feat_names:
            perm_sharpes = []
            for rep in range(n_repeats):
                rng = np.random.RandomState(rng_master.randint(0, 2**31 - 1))
                X_te = X_te_orig.copy()
                X_te[f] = rng.permutation(X_te[f].values)
                preds_perm = _train_and_predict_model(X_run, y_reg_run, train_inner_idx, cal_idx,
                                                     test_idx=None, X_test_df=X_te,
                                                     reg_params=reg_params, rank_params=rank_params, early_stopping_rounds=early_stopping_rounds)
                topk_perm = preds_perm.sort_values(ascending=False).head(top_k).index.tolist()
                cum_pf_perm, strat_ret_perm, pos_perm = simple_backtest_from_preds(topk_perm, df_run, HORIZON=HORIZON)
                adj_perm, _ = apply_roundtrip_costs_to_strat_returns(strat_ret_perm.fillna(0), topk_perm, df_run, roundtrip_cost=PRIMARY_COST)
                perm_sharpes.append(ann_stats_from_daily_returns(adj_perm)[2])
            perm_mean = np.nanmean(perm_sharpes)
            fold_importances[f].append(base_sharpe - perm_mean)
        n_folds += 1

    avg_importances = {f: (np.nanmean(vals) if len(vals)>0 else np.nan) for f, vals in fold_importances.items()}
    imp_series = pd.Series(avg_importances).sort_values(ascending=False)
    return imp_series, n_folds

# ------------------ RUNNERS (unchanged behavior) ------------------
def prepare_run_df_and_X(feature_cols):
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df_run = df.dropna(subset=feature_cols + ['future_return']).copy()
    X_run = df_run[feature_cols].copy()
    y_sign_run = (df_run['future_return'] > 0).astype(int)
    y_reg_run = df_run['future_return'].copy()
    return df_run, X_run, y_sign_run, y_reg_run

def run_full_pipeline_classification(feature_cols, run_name_suffix):
    df_run, X_run, y_sign_run, y_reg_run = prepare_run_df_and_X(feature_cols)
    all_results = []
    print(f"\n=== RUN: {run_name_suffix} (CLASSIFIER) -- calibrated classification k-sweep ===")
    for k in k_values:
        res, _, _, _, _ = run_calibrated_class_topk_for_X(X_run, y_sign_run, df_run, k, require_calibration=True, verbose=True)
        all_results.append(res)
    res_df = pd.DataFrame(all_results).sort_values('k').reset_index(drop=True)
    res_df.to_csv(f"results/k_sweep_classification_calibrated_results_{run_name_suffix}.csv", index=False)
    print("Done classification run.")
    return res_df

def run_full_pipeline_regression(feature_cols, run_name_suffix):
    df_run, X_run, y_sign_run, y_reg_run = prepare_run_df_and_X(feature_cols)
    all_results = []
    cum_pf_by_k = {}
    strat_ret_by_k = {}
    preds_by_k = {}
    start_time = time.time()
    print(f"\n=== RUN: {run_name_suffix} (REGRESSOR) -- walk-forward regressor k-sweep ===")
    for k in k_values:
        res, cum_pf_k, strat_ret_k, pred_dates_k, pred_all_k = run_regressor_topk_for_X(
            X_run, y_reg_run, df_run, k, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP, verbose=True)
        all_results.append(res)
        cum_pf_by_k[k] = cum_pf_k
        strat_ret_by_k[k] = strat_ret_k
        preds_by_k[k] = pred_all_k
        pd.DataFrame(all_results).to_csv(f"results/k_sweep_regressor_results_{run_name_suffix}_partial.csv", index=False)

    print("k-sweep finished in {:.1f}s".format(time.time() - start_time))
    res_df = pd.DataFrame(all_results).sort_values('k').reset_index(drop=True)
    res_df.to_csv(f"results/k_sweep_regressor_results_{run_name_suffix}.csv", index=False)
    print("\nK-sweep results (regressor) for run:", run_name_suffix)
    print(res_df[['k','n_trades','ann_return','ann_vol','sharpe','coverage_frac','exposure_frac']].to_string(index=False))

    # cost sweep
    cost_results = []
    for cost in cost_list:
        print(f"\nComputing net stats for round-trip cost = {cost:.6f} for run {run_name_suffix} ...")
        for k in k_values:
            res, cum_pf_k, strat_ret_k, pred_dates_k, pred_all_k = run_regressor_topk_for_X(
                X_run, y_reg_run, df_run, k, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP, verbose=False)
            adj_strat_ret, pos = apply_roundtrip_costs_to_strat_returns(strat_ret_k.fillna(0), pred_dates_k, df_run, roundtrip_cost=cost)
            ann_net = ann_stats_from_daily_returns(adj_strat_ret)
            cum_adj = (1 + adj_strat_ret).cumprod()
            cost_results.append({
                'cost_roundtrip': cost,
                'k': k,
                'n_trades': res['n_trades'],
                'ann_return_net': ann_net[0],
                'ann_vol_net': ann_net[1],
                'sharpe_net': ann_net[2],
                'cum_return_net': float(cum_adj.iloc[-1]) if len(cum_adj)>0 else np.nan,
                'coverage_frac': res['coverage_frac'],
                'exposure_frac': res['exposure_frac']
            })
            print(f" k={k} cost={cost:.4f} -> ann_ret_net={ann_net[0]:.4f}, sharpe_net={ann_net[2]:.4f}")

    cost_df = pd.DataFrame(cost_results)
    cost_df.to_csv(f"results/k_sweep_regressor_costs_{run_name_suffix}.csv", index=False)
    print(f"\nSaved cost sweep results to results/k_sweep_regressor_costs_{run_name_suffix}.csv")

    # choose best_k by net Sharpe at PRIMARY_COST
    primary_subset = cost_df[cost_df['cost_roundtrip'] == PRIMARY_COST]
    if primary_subset.shape[0] == 0:
        raise SystemExit(f"No results for PRIMARY_COST={PRIMARY_COST}. Check cost_list.")
    best_row = primary_subset.sort_values('sharpe_net', ascending=False).iloc[0]
    best_k_by_net_sharpe = int(best_row['k'])
    print(f"\nBest k by NET Sharpe at cost={PRIMARY_COST} for run {run_name_suffix}: k = {best_k_by_net_sharpe}")
    print("Best row (cost, k, ann_return_net, sharpe_net):")
    print(best_row[['cost_roundtrip','k','ann_return_net','sharpe_net']])

    # recompute final net equity for chosen k
    res_final, cum_pf_final, strat_ret_final, pred_dates_final, pred_all_final = run_regressor_topk_for_X(
        X_run, y_reg_run, df_run, best_k_by_net_sharpe, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP, verbose=False)
    adj_strat_ret_final, pos_final = apply_roundtrip_costs_to_strat_returns(strat_ret_final.fillna(0), pred_dates_final, df_run, roundtrip_cost=PRIMARY_COST)
    cum_adj_final = (1 + adj_strat_ret_final).cumprod()
    cum_adj_final.to_csv(f"results/equity_best_k_{best_k_by_net_sharpe}_{run_name_suffix}_netcost_{PRIMARY_COST:.6f}.csv", index=True)
    plt.figure(figsize=(10,6))
    plt.plot(cum_adj_final, label=f'Regressor best_k={best_k_by_net_sharpe} ({run_name_suffix}) net cost={PRIMARY_COST}')
    plt.plot((1 + df_run['Close'].pct_change().fillna(0)).cumprod(), label='Buy & Hold', alpha=0.7)
    plt.legend(); plt.title(f'Equity (regressor) best_k={best_k_by_net_sharpe} - {run_name_suffix}')
    plt.tight_layout()
    plt.savefig(f"results/equity_best_k_{best_k_by_net_sharpe}_{run_name_suffix}_netcost_{PRIMARY_COST:.6f}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # diagnostics - get full predicted series for threshold sweep & spearman
    pred_vals_all, _ = get_regressor_pred_series_for_X(X_run, y_reg_run, reg_params=REG_PARAMS, rank_params=RANKER_PARAMS, early_stopping_rounds=REG_EARLY_STOP)
    preds_nonan = pred_vals_all.dropna()
    spearman_rank = np.nan
    if preds_nonan.shape[0] > 0:
        aligned_real = df_run['future_return'].loc[preds_nonan.index]
        spearman_rank = preds_nonan.corr(aligned_real, method='spearman')
    print(f"Spearman rank corr (predicted_return vs realized future_return) [{run_name_suffix}]: {spearman_rank}")

    # ------------------ REPLACED THRESHOLD SWEEP FOR REGRESSOR (PERCENTILES) ------------------
    threshold_results = []
    pred_vals_all_nonan = pred_vals_all.dropna()
    adj_strat_ret_thr = pd.Series(0, index=df_run.index)  # fallback in case no preds

    if pred_vals_all_nonan.shape[0] == 0:
        print("Warning: no regressor predictions available for threshold sweep. Skipping percentile sweep.")
        threshold_df = pd.DataFrame([{
            'cost_roundtrip': PRIMARY_COST,
            'percentile_threshold': np.nan,
            'n_trades': 0,
            'ann_return_net': 0.0,
            'ann_vol_net': 0.0,
            'sharpe_net': 0.0,
            'cum_return_net': np.nan,
            'coverage_frac': 0.0,
            'exposure_frac': 0.0,
            'per_trade_mean': np.nan,
            'per_trade_winrate': np.nan
        }])
        best_th_row = threshold_df.iloc[0]
        best_p = float(best_th_row['percentile_threshold']) if pd.notna(best_th_row['percentile_threshold']) else np.nan
        preds_idx_final = []
    else:
        pct_grid = np.array([0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.50])
        ranks = pred_vals_all_nonan.rank(method='average', pct=True)

        for pct in pct_grid:
            preds_idx = ranks[ranks >= pct].index.tolist()
            cum_pf_p, strat_ret_p, pos_p = simple_backtest_from_preds(preds_idx, df_run, HORIZON=HORIZON)
            adj_strat_ret_p, pos_p = apply_roundtrip_costs_to_strat_returns(strat_ret_p.fillna(0), preds_idx, df_run, roundtrip_cost=PRIMARY_COST)
            ann_p = ann_stats_from_daily_returns(adj_strat_ret_p)
            entry_dates = preds_idx
            per_trade_returns = df_run.loc[entry_dates, 'future_return'].dropna() if len(entry_dates)>0 else pd.Series([], dtype=float)
            threshold_results.append({
                'cost_roundtrip': PRIMARY_COST,
                'percentile_threshold': float(pct),
                'n_trades': int(len(entry_dates)),
                'ann_return_net': ann_p[0],
                'ann_vol_net': ann_p[1],
                'sharpe_net': ann_p[2],
                'cum_return_net': float((1+adj_strat_ret_p).cumprod().iloc[-1]) if len(adj_strat_ret_p)>0 else np.nan,
                'coverage_frac': (len(entry_dates)/len(df_run)) if len(df_run)>0 else 0.0,
                'exposure_frac': float(pos_p.mean()) if hasattr(pos_p, 'mean') else np.nan,
                'per_trade_mean': float(per_trade_returns.mean()) if len(per_trade_returns)>0 else np.nan,
                'per_trade_winrate': float((per_trade_returns>0).mean()) if len(per_trade_returns)>0 else np.nan
            })

        threshold_df = pd.DataFrame(threshold_results).sort_values('percentile_threshold', ascending=False).reset_index(drop=True)
        threshold_df.to_csv(f"results/threshold_sweep_regressor_{run_name_suffix}_percentiles_primarycost_{PRIMARY_COST:.6f}.csv", index=False)

        best_th_row = threshold_df.sort_values('sharpe_net', ascending=False).iloc[0]
        best_p = float(best_th_row['percentile_threshold'])

        preds_idx_final = ranks[ranks >= best_p].index.tolist()
        cum_pf_thr, strat_ret_thr, pos_thr = simple_backtest_from_preds(preds_idx_final, df_run, HORIZON=HORIZON)
        adj_strat_ret_thr, pos_thr = apply_roundtrip_costs_to_strat_returns(strat_ret_thr.fillna(0), preds_idx_final, df_run, roundtrip_cost=PRIMARY_COST)
        cum_adj_thr = (1 + adj_strat_ret_thr).cumprod()
        cum_adj_thr.to_csv(f"results/equity_best_threshold_regressor_{run_name_suffix}_pct_{best_p:.4f}_netcost_{PRIMARY_COST:.6f}.csv", index=True)
        plt.figure(figsize=(10,6))
        plt.plot(cum_adj_thr, label=f'Regressor pct-threshold {best_p:.4f} ({run_name_suffix}) net cost={PRIMARY_COST}')
        plt.plot((1 + df_run['Close'].pct_change().fillna(0)).cumprod(), label='Buy & Hold', alpha=0.7)
        plt.legend(); plt.title(f'Equity (regressor pct-th={best_p:.4f}) - {run_name_suffix}')
        plt.tight_layout()
        plt.savefig(f"results/equity_best_threshold_regressor_{run_name_suffix}_pct_{best_p:.4f}_netcost_{PRIMARY_COST:.6f}.png", dpi=150, bbox_inches='tight')
        plt.close()

    try:
        ann_thr_final = ann_stats_from_daily_returns(adj_strat_ret_thr)
    except Exception:
        ann_thr_final = (0.0, 0.0, 0.0)

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    model_daily_thr = adj_strat_ret_thr.fillna(0).reset_index(drop=True)
    bh_daily = df_run['Close'].pct_change().fillna(0).reset_index(drop=True)
    n = len(model_daily_thr)
    diffs_thr = []
    for i in range(BOOTSTRAP_N):
        if n == 0:
            diffs_thr.append(0.0)
            continue
        idx = rng.randint(0, n, n)
        sample_model = model_daily_thr.iloc[idx].reset_index(drop=True)
        sample_bh = bh_daily.iloc[idx].reset_index(drop=True)
        s_model = ann_stats_from_daily_returns(sample_model)[2]
        s_bh = ann_stats_from_daily_returns(sample_bh)[2]
        diffs_thr.append(s_model - s_bh)
    diffs_thr = np.array(diffs_thr)
    pval_thr = np.mean(np.abs(diffs_thr) >= abs(ann_thr_final[2] - ann_stats_from_daily_returns(df_run['Close'].pct_change().fillna(0))[2])) if len(diffs_thr)>0 else np.nan

    out_summary = {
        'run_name': run_name_suffix,
        'k_sweep_df': res_df.to_dict(orient='list') if 'res_df' in locals() else None,
        'best_k_by_net_sharpe': best_k_by_net_sharpe,
        'best_k_row': best_row.to_dict() if 'best_row' in locals() else None,
        'best_threshold_p': best_p if 'best_p' in locals() else None,
        'threshold_row': best_th_row.to_dict() if 'best_th_row' in locals() else None,
        'ann_net_final': ann_stats_from_daily_returns(adj_strat_ret_final) if 'adj_strat_ret_final' in locals() else (0.0,0.0,0.0),
        'ann_thr_final': ann_thr_final,
        'ann_bh': ann_stats_from_daily_returns(df_run['Close'].pct_change().fillna(0)),
        'spearman_rank': spearman_rank,
        'pval_threshold_bootstrap': float(pval_thr) if not pd.isna(pval_thr) else None
    }
    pd.DataFrame([out_summary]).to_json(f"results/summary_regressor_{run_name_suffix}.json", orient='records', lines=False)
    pd.DataFrame({'diffs': diffs_thr}).to_csv(f"results/bootstrap_best_threshold_regressor_{run_name_suffix}_diffs.csv", index=False)

    print(f"\nRun {run_name_suffix} (REGRESSOR) DONE. summary saved to results/summary_regressor_{run_name_suffix}.json")
    return out_summary

# ------------------ FINAL MODEL SAVE (NEW) ------------------
def train_and_save_final_model(feature_cols, ticker, mode, out_dir="results"):
    """
    Train a final model on ALL available history (df prepared for ticker must be global `df`)
    using the provided feature_cols and mode ('RANKER' or 'REGRESSOR'), then save model + metadata.
    - Saves: results/final_model_{ticker}.pkl
             results/model_metadata_{ticker}.json
    """
    global df
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_run, X_run, y_sign_run, y_reg_run = prepare_run_df_and_X(feature_cols)
    except Exception as e:
        print(f"Could not prepare run df for final model save for {ticker}: {e}")
        return None

    if X_run.shape[0] < max(30, min_train_size):
        print(f"Not enough rows to train final model for {ticker}: {X_run.shape[0]} rows.")
        return None

    # Clone params locally
    if mode == "RANKER":
        params = copy.deepcopy(RANKER_PARAMS)
        # create labels across whole history similar to walk-forward ranker label creation
        try:
            y_clip = y_reg_run.copy()
            if TARGET_CLIP_MODE == 'quantile':
                low_q, high_q = TARGET_CLIP_QUANTILES
                lo = y_clip.quantile(low_q)
                hi = y_clip.quantile(high_q)
                y_clip = y_clip.clip(lo, hi)
            else:
                lo, hi = TARGET_CLIP_ABS
                y_clip = y_clip.clip(lo, hi)

            ranked = y_clip.rank(method='first')
            n_bins = min(RANKER_N_BINS, max(2, int(len(ranked) // 2)))
            labels = pd.qcut(ranked, q=n_bins, labels=False, duplicates='drop').astype(int)
        except Exception as e:
            print("Warning: ranker label creation failed, using binary ranks fallback:", e)
            labels = (y_clip.rank(method='first') > y_clip.rank(method='first').median()).astype(int)

        groups = _groups_from_index(X_run.index)
        model = LGBMRanker(**params)
        try:
            print(f"Training final RANKER on full history for {ticker} with {len(X_run)} rows...")
            model.fit(X_run, labels, group=groups)
        except Exception as e:
            try:
                print("Retrying ranker.fit without group...")
                model.fit(X_run, labels)
            except Exception as e2:
                print("Final ranker training failed:", e2)
                return None

    else:
        # REGRESSOR
        params = copy.deepcopy(REG_PARAMS)
        y_clip = y_reg_run.copy()
        if TARGET_CLIP_MODE == 'quantile':
            low_q, high_q = TARGET_CLIP_QUANTILES
            lo = y_clip.quantile(low_q)
            hi = y_clip.quantile(high_q)
            y_clip = y_clip.clip(lo, hi)
        else:
            lo, hi = TARGET_CLIP_ABS
            y_clip = y_clip.clip(lo, hi)

        model = LGBMRegressor(**params)
        try:
            print(f"Training final REGRESSOR on full history for {ticker} with {len(X_run)} rows...")
            model.fit(X_run, y_clip)
        except Exception as e:
            print("Final regressor training failed:", e)
            return None

    # save model
    model_path = out_dir / f"final_model_{ticker.replace('/','_')}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved final model -> {model_path}")

    # Build metadata and save
    meta = {
        "feature_list": feature_cols,
        "HORIZON": int(HORIZON),
        "model_type": "ranker" if mode == "RANKER" else "regressor",
        "train_end_date": str(df_run.index.max().date()) if hasattr(df_run.index, 'max') else None,
        "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
        "notes": f"Final model trained on full history for {ticker} using TOP_{len(feature_cols)} features",
        "version": "v1"
    }
    meta_path = out_dir / f"model_metadata_{ticker.replace('/','_')}.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata -> {meta_path}")

    return {"model_path": str(model_path), "meta_path": str(meta_path)}

# ------------------ CROSS-TICKER RUNNER ------------------
def run_full_sequence_for_ticker(ticker, perm_repeats_for_permimp=None):
    """
    Sets up global df for the ticker, then runs the main pipeline steps in the same order
    as original script. If perm_repeats_for_permimp is provided, uses that for the
    walkforward permutation importance call (speeds it up).
    After selecting top features and running selected run, trains & saves a final model
    on full history using those selected features.
    """
    global df, TICKER
    TICKER = ticker
    try:
        df = prepare_df_for_ticker(ticker)
    except Exception as e:
        print(f"Failed to prepare data for {ticker}: {e}")
        return {'ticker': ticker, 'error': str(e)}

    # run baseline classifier
    summary_baseline_clf = run_full_pipeline_classification(baseline_feature_cols, f"{ticker}_BASELINE")

    # run engineered classifier
    summary_engineered_clf = run_full_pipeline_classification(engineered_feature_cols, f"{ticker}_ENGINEERED")

    mode = "RANKER" if USE_RANKER else "REGRESSOR"
    print(f"\nRunning ENGINEERED_{mode} (model on full engineered features) for {ticker}...")
    summary_engineered_reg = run_full_pipeline_regression(engineered_feature_cols, f"ENGINEERED_{mode}_{ticker}")

    # Permutation importance: optionally use a reduced n_repeats (only for speed during cross-ticker)
    use_n_repeats = PERM_N_REPEATS if perm_repeats_for_permimp is None else perm_repeats_for_permimp
    print("\nComputing walk-forward permutation importance (regressor/ranker) on ENGINEERED features...")
    df_eng, X_eng, y_sign_eng, y_reg_eng = prepare_run_df_and_X(engineered_feature_cols)
    imp_series, n_folds = walkforward_permutation_importance_regressor(df_eng, X_eng, y_reg_eng,
                                                                        engineered_feature_cols,
                                                                        top_k=PERM_TOP_K,
                                                                        n_repeats=use_n_repeats,
                                                                        random_seed=PERM_RANDOM_SEED,
                                                                        reg_params=REG_PARAMS,
                                                                        rank_params=RANKER_PARAMS,
                                                                        early_stopping_rounds=REG_EARLY_STOP)
    print(f"Permutation importance computed over {n_folds} folds.")
    imp_series.to_csv(f"results/feature_importances_walkforward_ENGINEERED_regressor_ranker_fixed_{ticker}.csv", header=['importance'])
    plt.figure(figsize=(12,6))
    imp_series.plot(kind='bar')
    plt.title(f"Walk-forward permutation importance (avg drop in NET-Sharpe) - ENGINEERED ({ticker})")
    plt.ylabel("Avg importance (sharpe drop)")
    plt.tight_layout()
    plt.savefig(f"results/feature_importances_walkforward_ENGINEERED_regressor_ranker_fixed_{ticker}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nTop features (by permutation importance):")
    print(imp_series.head(TOP_N).to_string())

    selected_features = imp_series.head(TOP_N).index.tolist()
    print(f"\nSelected TOP_N={TOP_N} features for final model run on {ticker}:", selected_features)
    summary_selected_reg = run_full_pipeline_regression(selected_features, f"SELECTED_TOP{TOP_N}_{mode}_{ticker}")

    # Train & save final model on full history using selected features
    saved_info = train_and_save_final_model(selected_features, ticker, mode, out_dir="results")
    if saved_info is not None:
        print(f"Final model & metadata saved for {ticker}: {saved_info}")
    else:
        print(f"Final model save skipped/failed for {ticker}.")

    # Load summary JSONs (if present) and return a compact summary dict for the cross-ticker CSV
    try:
        with open(f"results/summary_regressor_ENGINEERED_{mode}_{ticker}.json", 'r') as fh:
            eng_sum = json.load(fh)
            if isinstance(eng_sum, list) and len(eng_sum) > 0:
                eng_sum = eng_sum[0]
    except Exception:
        eng_sum = None
    try:
        with open(f"results/summary_regressor_SELECTED_TOP{TOP_N}_{mode}_{ticker}.json", 'r') as fh:
            sel_sum = json.load(fh)
            if isinstance(sel_sum, list) and len(sel_sum) > 0:
                sel_sum = sel_sum[0]
    except Exception:
        sel_sum = None

    return {'ticker': ticker, 'engineered_summary': eng_sum, 'selected_summary': sel_sum}

# ------------------ MAIN (cross-ticker orchestration) ------------------
if __name__ == "__main__":
    t0 = time.time()

    cross_results = []
    for tk in CROSS_TICKERS:
        print("\n" + "="*80)
        print(f"STARTING PIPELINE FOR {tk}")
        print("="*80)
        # For cross-ticker runs we speed up permutation importance by lowering repeats
        res = run_full_sequence_for_ticker(tk, perm_repeats_for_permimp=PERM_REPEATS_FOR_CROSS_TICKER)
        cross_results.append(res)
        # flush to disk after each ticker
        pd.DataFrame([res]).to_json(f"results/cross_run_partial_{tk}.json", orient='records', lines=False)

    # Build an aggregate CSV summary
    rows = []
    for r in cross_results:
        if r.get('error'):
            rows.append({'ticker': r['ticker'], 'error': r['error']})
            continue
        eng = r.get('engineered_summary') or {}
        sel = r.get('selected_summary') or {}
        rows.append({
            'ticker': r['ticker'],
            'engineered_best_k': eng.get('best_k_by_net_sharpe') if isinstance(eng, dict) else None,
            'engineered_best_net_sharpe': eng.get('best_k_row', {}).get('sharpe_net') if isinstance(eng, dict) and eng.get('best_k_row') else None,
            'engineered_ann_net_final_sharpe': (eng.get('ann_net_final')[2] if (isinstance(eng, dict) and eng.get('ann_net_final')) else None),
            'engineered_spearman': eng.get('spearman_rank') if isinstance(eng, dict) else None,
            'selected_best_k': sel.get('best_k_by_net_sharpe') if isinstance(sel, dict) else None,
            'selected_best_net_sharpe': sel.get('best_k_row', {}).get('sharpe_net') if isinstance(sel, dict) and sel.get('best_k_row') else None,
            'selected_ann_net_final_sharpe': (sel.get('ann_net_final')[2] if (isinstance(sel, dict) and sel.get('ann_net_final')) else None),
            'selected_spearman': sel.get('spearman_rank') if isinstance(sel, dict) else None,
            'selected_best_threshold_p': sel.get('best_threshold_p') if isinstance(sel, dict) else None,
            'selected_pval_threshold_bootstrap': sel.get('pval_threshold_bootstrap') if isinstance(sel, dict) else None
        })

    pd.DataFrame(rows).to_csv("results/cross_ticker_summary.csv", index=False)
    print("\nCross-ticker run finished. Summary saved to results/cross_ticker_summary.csv")
    print(f"Total elapsed time: {time.time() - t0:.1f}s")

