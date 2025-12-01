#!/usr/bin/env python3
# predict_signals.py
"""
Interactive signal generator with diagnostics that loads final_model_{TICKER}.pkl
and model_metadata_{TICKER}.json (saved by your training script).

If a required argument is not passed on the command line, the script will prompt the user interactively.
Enhancements:
 - prints latest raw market date (market_last)
 - prints latest scored date (scored_last)
 - lists missing features on the most recent market rows when scoring cannot include them
 - compares metadata train_end_date vs expected train_end_date (based on HORIZON shift)
"""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import sys
import datetime

# Import your feature builder from the training script (adjust name if needed)
try:
    from run_momentum_classifier import prepare_df_for_ticker
except Exception as e:
    print("ERROR: could not import prepare_df_for_ticker from training script.")
    print("Make sure run_momentum_classifier.py is in the same folder and contains prepare_df_for_ticker.")
    raise

def prompt_with_default(prompt_text, default=None, allow_empty=False):
    if default is None:
        display = ""
    else:
        display = f" [{default}]"
    while True:
        try:
            val = input(f"{prompt_text}{display}: ").strip()
        except EOFError:
            return default
        if val == "":
            if default is not None:
                return str(default)
            if allow_empty:
                return ""
            print("Please enter a value (or press Ctrl+C to quit).")
            continue
        return val

def load_model_and_meta(model_path: Path, meta_path: Path):
    model = joblib.load(str(model_path))
    with open(str(meta_path), "r") as fh:
        meta = json.load(fh)
    return model, meta

def score_df_with_model(df: pd.DataFrame, model, feature_list: list, model_type: str):
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat.index = pd.to_datetime(df_feat.index)
    df_feat = df_feat.dropna(subset=feature_list).copy()
    if df_feat.shape[0] == 0:
        return df_feat
    X = df_feat[feature_list]
    try:
        if model_type == "classifier":
            try:
                probs = model.predict_proba(X)
                scores = probs[:, 1]
            except Exception:
                scores = model.predict(X)
        else:
            scores = model.predict(X)
    except Exception:
        scores = model.predict(X)
    df_feat["score"] = scores
    if model_type in ("regressor", "ranker"):
        df_feat["pred_return"] = df_feat["score"]
    return df_feat

def monthly_topk_signal(df_scores: pd.DataFrame, date: pd.Timestamp, k: int):
    period = date.to_period("M")
    month_idx = df_scores.index.to_series().dt.to_period("M") == period
    month_scores = df_scores[month_idx].sort_values("score", ascending=False)
    if month_scores.shape[0] == 0:
        return None, None
    topk = month_scores.head(k)
    is_buy = date in topk.index
    try:
        rank = (month_scores.index.get_loc(date) + 1) if date in month_scores.index else None
    except Exception:
        rank = None
    return bool(is_buy), rank

def percentile_signal(df_scores: pd.DataFrame, date: pd.Timestamp, percentile_threshold: float):
    if df_scores.shape[0] == 0:
        return None, None
    today_score = float(df_scores.loc[date, "score"])
    pct = (df_scores["score"] < today_score).mean() * 100.0
    top_pct = 100.0 - percentile_threshold
    is_buy = pct >= top_pct
    return bool(is_buy), pct

def append_signal_log(out_csv: Path, record: dict):
    df = pd.DataFrame([record])
    header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, mode="a", header=header)

def print_signal_block(ticker, date, record):
    print(f"\n=== Signal for {ticker} on {date.date()} ===")
    print(f"Model version: {record.get('model_version','n/a')}   Trained up to (metadata): {record.get('train_end_date','n/a')}")
    print(f"Score: {record.get('score'):.6f}   PredReturn(est): {record.get('pred_return') if record.get('pred_return') is not None else 'n/a'}")
    if record.get('topk_signal') is not None:
        print(f"Top-k (k={record.get('k_used')}): {'BUY' if record['topk_signal'] else 'NO'}   (month-rank: {record.get('topk_rank')})")
    if record.get('percentile_signal') is not None:
        top_pct = 100.0 - record.get('percentile_used')
        print(f"Percentile rule (top {top_pct:.2f}%): {'BUY' if record['percentile_signal'] else 'NO'}   (score percentile: {record.get('score_percentile'):.2f})")
    print("-"*48)

def missing_features_on_dates(df: pd.DataFrame, feature_list: list, dates):
    """
    For each date in dates (index), return list of features that are NaN on that date.
    """
    out = {}
    for d in dates:
        if d not in df.index:
            out[str(d.date())] = ["<date not in raw df>"]
            continue
        row = df.loc[d, feature_list]
        # If single-index selection returns Series; if feature_list length ==1 returns scalar - handle general
        if isinstance(row, pd.Series):
            missing = row[row.isna()].index.tolist()
        else:
            # scalar or df row as scalar -> coerce
            missing = [feature_list[0]] if pd.isna(row) else []
        out[str(d.date())] = missing
    return out

def main(args):
    # interactive prompts for missing args
    if not args.tickers:
        s = prompt_with_default("Enter tickers (comma-separated, e.g. BHARTIARTL.NS,RELIANCE.NS)", None)
        if not s:
            print("No tickers provided. Exiting.")
            sys.exit(1)
        args.tickers = [t.strip() for t in s.split(",") if t.strip()]

    if not args.model_dir:
        args.model_dir = prompt_with_default("Enter model directory where final_model_{TICKER}.pkl lives", "results")

    if args.k is None:
        k_in = prompt_with_default("Enter monthly top-k (enter 0 to disable)", "3")
        try:
            k_val = int(k_in)
            args.k = None if k_val == 0 else k_val
        except Exception:
            args.k = 3

    if args.percentile is None:
        pct_in = prompt_with_default("Enter percentile (e.g. 99.5 means top 0.5%) (enter 0 to disable)", "99.5")
        try:
            pct_val = float(pct_in)
            args.percentile = None if pct_val == 0 else pct_val
        except Exception:
            args.percentile = 99.5

    if not args.out_csv:
        args.out_csv = prompt_with_default("Enter output CSV path", "results/signals_log.csv")

    if not args.date:
        date_in = prompt_with_default("Enter date to score (YYYY-MM-DD) or press Enter for latest", "", allow_empty=True)
        args.date = date_in if date_in != "" else None

    if args.horizon is None:
        horizon_in = prompt_with_default("Enter HORIZON fallback (days)", "5")
        try:
            args.horizon = int(horizon_in)
        except Exception:
            args.horizon = 5

    model_dir = Path(args.model_dir)
    out_csv = Path(args.out_csv)

    for tk in args.tickers:
        model_path = model_dir / f"final_model_{tk.replace('/','_')}.pkl"
        meta_path = model_dir / f"model_metadata_{tk.replace('/','_')}.json"

        if not model_path.exists() or not meta_path.exists():
            print(f"[WARN] Missing model or metadata for {tk}.\n  {model_path}\n  {meta_path}\nSkipping.")
            continue

        model, meta = load_model_and_meta(model_path, meta_path)
        feature_list = meta.get("feature_list", None)
        model_type = meta.get("model_type", "ranker")
        train_end_date_meta = meta.get("train_end_date", None)
        model_version = meta.get("version", None)
        meta_horizon = int(meta.get("HORIZON", args.horizon))

        if not feature_list:
            print(f"[WARN] No 'feature_list' in metadata for {tk}. Skipping.")
            continue

        # build df with latest available OHLCV + features
        try:
            df = prepare_df_for_ticker(tk)
        except Exception as e:
            print(f"[ERROR] prepare_df_for_ticker failed for {tk}: {e}")
            continue

        df = df.sort_index()

        # raw latest market date (what yfinance returned as newest index)
        market_last = df.index.max() if len(df) > 0 else None
        print(f"\nDownloaded data for {tk}. Latest market row = {market_last.date() if market_last is not None else 'n/a'}")

        # Compute expected train_end_date based on how training script builds df_run
        try:
            df_run_candidate = df.dropna(subset=feature_list + ['future_return']).copy()
            train_end_expected = df_run_candidate.index.max() if len(df_run_candidate) > 0 else None
        except Exception:
            train_end_expected = None

        if train_end_date_meta is not None:
            print(f"Model metadata train_end_date: {train_end_date_meta}")
        if train_end_expected is not None:
            print(f"Expected train_end_date (based on current raw data and HORIZON={meta_horizon}): {train_end_expected.date()}")
        else:
            print("Expected train_end_date: could not compute (insufficient data)")

        if train_end_date_meta != (str(train_end_expected.date()) if train_end_expected is not None else None):
            print("NOTE: metadata train_end_date differs from expected value based on current raw data.")
            print(" This is normal if you trained the model earlier and then downloaded new market data since training.")
            print(" If you want the model trained further you must re-run the training pipeline (final model save) on updated data.")
        else:
            print("Metadata train_end_date matches expected date based on the data used to save model.")

        # Score
        df_scores = score_df_with_model(df, model, feature_list, model_type)

        if df_scores.shape[0] == 0:
            print(f"[WARN] After dropping NaNs no rows remain with all features for {tk}.")
            # show which features are missing for the most recent market date(s)
            if market_last is not None:
                check_dates = [market_last] + list(df.index[-5:])  # show last few
                miss = missing_features_on_dates(df, feature_list, sorted(set(check_dates)))
                print("Missing features at recent dates (date -> missing features):")
                for d, feats in miss.items():
                    print(f"  {d} -> {feats}")
            else:
                print("No market data available.")
            continue

        scored_last = df_scores.index.max()

        # compare market_last vs scored_last
        print(f"Latest scored row (all required features present) = {scored_last.date()}")
        if market_last is not None and scored_last < market_last:
            print("Note: latest scored row is earlier than the latest market row — this happens when some features are NaN for the newest market rows.")
            # show missing features on the latest market rows (market_last and few prior)
            check_dates = [market_last] + list(df.index[-(meta_horizon+1):])
            check_dates = sorted(set([d for d in check_dates if d in df.index]))
            miss = missing_features_on_dates(df, feature_list, check_dates)
            print("Missing features at recent dates (date -> missing features):")
            for d, feats in miss.items():
                print(f"  {d} -> {feats}")
            print("If these are NaNs due to corporate actions or stale data, re-download / fix the raw data or wait for the next trading day.")
        else:
            print("Latest scored row equals latest market row (good).")

        # choose the "today" to report on
        today = scored_last
        if args.date is not None:
            try:
                requested = pd.to_datetime(args.date)
                if requested in df_scores.index:
                    today = requested
                else:
                    earlier = df_scores.index[df_scores.index <= requested]
                    if len(earlier) == 0:
                        print(f"[WARN] requested date {requested.date()} not present and no earlier scored rows. Using latest scored date.")
                    else:
                        today = earlier.max()
            except Exception:
                print("[WARN] couldn't parse --date argument; using latest scored date.")
                today = scored_last

        rec = {
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "ticker": tk,
            "date": str(today.date()),
            "model_version": model_version,
            "train_end_date": train_end_date_meta,
            "model_type": model_type,
            "HORIZON": int(meta.get("HORIZON", args.horizon)),
            "score": float(df_scores.loc[today, "score"]),
            "pred_return": float(df_scores.loc[today, "pred_return"]) if "pred_return" in df_scores.columns else None
        }

        # monthly top-k rule
        if args.k is not None:
            topk_sig, rank = monthly_topk_signal(df_scores, today, args.k)
            rec.update({
                "topk_signal": topk_sig,
                "topk_rank": rank,
                "k_used": args.k
            })
        else:
            rec.update({"topk_signal": None, "topk_rank": None, "k_used": None})

        # percentile rule
        if args.percentile is not None:
            pct_sig, pct_val = percentile_signal(df_scores, today, args.percentile)
            rec.update({
                "percentile_signal": pct_sig,
                "score_percentile": float(pct_val) if pct_val is not None else None,
                "percentile_used": float(args.percentile)
            })
        else:
            rec.update({"percentile_signal": None, "score_percentile": None, "percentile_used": None})

        # print nicely
        print_signal_block(tk, today, rec)

        # append to CSV
        append_signal_log(out_csv, rec)

    print(f"\nAll done. Signals appended to {out_csv.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--tickers", nargs="+", help="Tickers to score (e.g. BHARTIARTL.NS). Comma separated when prompted.")
    p.add_argument("--model_dir", help="Directory where final_model_{TICKER}.pkl and metadata live.")
    p.add_argument("--k", type=int, help="monthly top-k value (0 to disable).")
    p.add_argument("--percentile", type=float, help="percentile threshold (e.g. 99.5). 0 to disable.")
    p.add_argument("--out_csv", help="CSV file to append signal records.")
    p.add_argument("--date", help="Optional: force a date (YYYY-MM-DD) to score instead of latest available.")
    p.add_argument("--horizon", type=int, help="HORIZON fallback if missing in metadata.")
    p.add_argument("--help", action="store_true", help="Show help and exit.")
    args, unknown = p.parse_known_args()

    if args.help:
        print("\nThis script accepts command-line args or runs interactively to get missing inputs.\n")
        print("Arguments (CLI):")
        print("  --tickers TICKER [TICKER ...]    one or more tickers e.g. BHARTIARTL.NS")
        print("  --model_dir DIR                  directory with final_model_{TICKER}.pkl (default: results)")
        print("  --k INT                          monthly top-k (0 to disable, default: 3)")
        print("  --percentile FLOAT               percentile threshold e.g. 99.5 (0 to disable, default: 99.5)")
        print("  --out_csv PATH                   file to append signals (default: results/signals_log.csv)")
        print("  --date YYYY-MM-DD                optional explicit date to score (default: latest available)")
        print("  --horizon INT                    fallback horizon (default: 5)\n")
        sys.exit(0)

    main(args)
