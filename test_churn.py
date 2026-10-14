"""Leakage check: features see only the past window, the label sees only the 90 days after the cutoff. Run: python test_churn.py"""
import pandas as pd

from train import snapshot

CUTOFF = pd.Timestamp("2011-06-01")


def tx(rows):
    df = pd.DataFrame(rows, columns=["customer", "days", "value"])
    return pd.DataFrame({
        "customer": df.customer, "invoice": [f"{'C' if v < 0 else ''}{i}" for i, v in enumerate(df.value)],
        "date": CUTOFF + pd.to_timedelta(df.days, unit="D"), "stock": "X", "value": df.value,
        "country": "United Kingdom", "is_return": df.value < 0,
    })


def test_snapshot():
    rows = [
        (1, -10, 100), (1, -40, 100), (1, 30, 999),  # buys again inside the horizon
        (2, -10, 100), (2, 90, 100),                  # next order on day 90: just outside [cutoff, cutoff + 90)
        (3, -200, 100),                               # last order before the 180-day window: not an active customer
        (4, -5, 100), (4, 10, -50),                   # only a return after the cutoff: still churned
        (5, 0, 100),                                  # first order on the cutoff day itself counts as future
    ]
    X = snapshot(tx(rows), CUTOFF)
    assert sorted(X.index) == [1, 2, 4], X.index
    assert X.churned.to_dict() == {1: 0, 2: 1, 4: 1}, X.churned
    # Post-cutoff rows (the 999 order, the return) must not change any feature.
    past_only = snapshot(tx([r for r in rows if r[1] < 0] + [(9, 95, 1)]), CUTOFF)  # (9) keeps the horizon complete
    features = X.columns.drop("churned")
    pd.testing.assert_frame_equal(X[features], past_only.loc[X.index, features])
    assert X.loc[1, "recency"] == 10 and X.loc[1, "orders_180"] == 2 and X.loc[1, "avg_gap"] == 30
    assert X.loc[4, "return_rate"] == 0


if __name__ == "__main__":
    test_snapshot()
    print("ok")
