"""Score active customers at the dataset's end: churn probability, RFM segment and contact priority.

Run after train.py: python score.py   (writes results/scores.csv)
"""
import joblib
import pandas as pd

from train import FEATURES, RFM, ROOT, load, snapshot


def main():
    saved = joblib.load(ROOT / "models" / "churn.joblib")
    tx = load()
    today = tx.date.max().normalize() + pd.Timedelta(days=1)  # day after the last transaction
    X = snapshot(tx, today)
    X["p_churn"] = saved["model"].predict_proba(X[FEATURES])[:, 1]
    X["at_risk"] = (X.p_churn >= saved["threshold"]).astype(int)
    X["segment"] = pd.Series(saved["segmenter"].predict(X[RFM]), X.index).map(saved["segment_names"])
    X["spend_at_risk"] = X.p_churn * X.spend_180  # priority score, not an estimate of future lost revenue
    out = X.sort_values("spend_at_risk", ascending=False)[
        ["segment", "p_churn", "at_risk", "recency", "orders_180", "spend_180", "spend_at_risk"]].round(3)
    out.to_csv(ROOT / "results" / "scores.csv", index_label="customer")

    print(f"Scored {len(X):,} customers active in the 6 months before {today.date()} with {saved['name']}")
    print(out.groupby("segment").agg(customers=("p_churn", "size"), at_risk=("at_risk", "sum"),
                                     avg_p_churn=("p_churn", "mean"), spend_at_risk=("spend_at_risk", "sum"))
          .sort_values("spend_at_risk", ascending=False).round(2).to_string())
    print("\nCall first: flagged customers ranked by probability-weighted historical spend\n", out[out.at_risk == 1].head(10).to_string())


if __name__ == "__main__":
    main()
