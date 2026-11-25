"""Customer churn: build leakage-safe snapshots, compare 3 models against a business rule, segment customers with RFM + K-means.

Run: python train.py   (writes models/churn.joblib and results/)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HORIZON = pd.Timedelta(days=90)  # churned = no purchase in the 90 days after the cutoff
WINDOW = pd.Timedelta(days=180)  # features look back 6 months; "active" = bought in that window
# Fees, postage, manual adjustments and test lines: not products a customer chose to buy.
NON_PRODUCT = {"POST", "DOT", "C2", "C3", "M", "m", "D", "S", "B", "BANK CHARGES", "AMAZONFEE",
               "ADJUST", "ADJUST2", "CRUK", "TEST001", "TEST002"}
FEATURES = ["recency", "tenure", "orders_90", "orders_180", "spend_180", "avg_order_value",
            "products_180", "avg_gap", "return_rate", "uk", "peak_season"]
RFM = ["recency", "orders_180", "spend_180"]
K = 4  # silhouette peaks at k=2, but that only splits active from lapsed; k=4 scores best among k >= 3
TOP = 0.2  # retention team capacity: contact the riskiest 20%


def load():
    cache = DATA / "online_retail_ii.csv"
    if not cache.exists():  # reading the xlsx takes ~45 s, so do it once
        sheets = pd.read_excel(DATA / "online_retail_II.xlsx", sheet_name=None)
        pd.concat(sheets.values()).to_csv(cache, index=False)
    df = pd.read_csv(cache, parse_dates=["InvoiceDate"], dtype={"Invoice": str, "StockCode": str})
    df = df.drop_duplicates()  # 1-9 Dec 2010 is in both sheets, plus double-scanned lines
    df = df[df["Customer ID"].notna() & ~df.StockCode.isin(NON_PRODUCT) & (df.Price > 0)]
    return pd.DataFrame({
        "customer": df["Customer ID"].astype(int), "invoice": df.Invoice, "date": df.InvoiceDate,
        "stock": df.StockCode, "value": df.Quantity * df.Price, "country": df.Country,
        "is_return": df.Invoice.str.startswith("C"),  # cancellations/returns carry negative quantity
    })


def snapshot(tx, cutoff):
    """One row per customer who bought in the WINDOW before `cutoff`.
    Features use only that window, so every snapshot measures them the same way;
    the label uses only the HORIZON after the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    recent = tx[(tx.date < cutoff) & (tx.date >= cutoff - WINDOW)]
    buys = recent[~recent.is_return]
    g = buys.groupby("customer")
    first_seen = tx[tx.date < cutoff].groupby("customer").date.min()
    X = pd.DataFrame({
        "recency": (cutoff - g.date.max()).dt.days,
        "tenure": (cutoff - first_seen).dt.days.clip(upper=WINDOW.days),  # capped: early snapshots have less history
        "orders_90": buys[buys.date >= cutoff - HORIZON].groupby("customer").invoice.nunique(),
        "orders_180": g.invoice.nunique(),
        "spend_180": g.value.sum(),
        "products_180": g.stock.nunique(),
        "avg_gap": (g.date.max() - g.date.min()).dt.days / (g.invoice.nunique() - 1),  # NaN for one-order customers
        "returned": -recent[recent.is_return].groupby("customer").value.sum(),
        "uk": (g.country.last() == "United Kingdom").astype(int),
    }).loc[lambda d: d.orders_180.notna()]
    X[["orders_90", "returned"]] = X[["orders_90", "returned"]].fillna(0)
    X["return_rate"] = (X.returned / X.spend_180).clip(0, 1)
    X["spend_180"] = (X.spend_180 - X.returned).clip(lower=0)  # net: some huge orders were cancelled the same day
    X["avg_order_value"] = X.spend_180 / X.orders_180
    # Churn is seasonal (gift wholesaler): a horizon covering the Nov-Dec rush has far less churn.
    X["peak_season"] = int(cutoff.month in (8, 9, 10))
    if cutoff + HORIZON <= tx.date.max():
        future = tx[(tx.date >= cutoff) & (tx.date < cutoff + HORIZON) & ~tx.is_return]
        X["churned"] = (~X.index.isin(future.customer)).astype(int)
    else:
        X["churned"] = np.nan  # scoring date: the future isn't known yet
    X["cutoff"] = cutoff
    return X.drop(columns="returned")


def models():
    impute = lambda: SimpleImputer(strategy="median", add_indicator=True)  # flags one-order customers
    return {
        "logreg": make_pipeline(impute(), FunctionTransformer(np.log1p), StandardScaler(),
                                LogisticRegression(max_iter=1000)),
        "random_forest": make_pipeline(impute(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=0)),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0),
    }


def best_f1_threshold(y, p):
    precision, recall, t = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return t[np.argmax(f1[:-1])]


def scores(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    yhat = p >= threshold
    top = np.argsort(-p, kind="stable")[: int(len(p) * TOP)]
    return {"roc_auc": roc_auc_score(y, p), "f1": f1_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "accuracy": accuracy_score(y, yhat),
            f"precision_top{TOP:.0%}": y[top].mean()}


def segmenter():
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), KMeans(K, n_init=10, random_state=0))


def name_segments(seg, X):
    """Name the K=4 clusters from their medians, so names don't depend on K-means' arbitrary cluster ids."""
    med = X[RFM].groupby(seg.predict(X[RFM])).median()
    names = {med.spend_180.idxmax(): "Champions", med.recency.idxmax(): "Lapsing"}
    rest = med.drop(names).sort_values("recency")
    names |= dict(zip(rest.index, ["Recent, low spend", "Regulars"]))
    return names


def split(tx):
    """Out-of-time split. Test = the last cutoff whose 90-day label window is complete. Train = every earlier
    quarter with a full feature window (their label windows all end by the test cutoff), which includes
    the same season a year earlier."""
    test_cutoff = tx.date.max().normalize() - HORIZON
    starts = [test_cutoff - HORIZON * k for k in range(1, 9) if test_cutoff - HORIZON * k - WINDOW >= tx.date.min()]
    return pd.concat([snapshot(tx, c) for c in starts]), snapshot(tx, test_cutoff)


def main():
    train, test = split(load())
    test_cutoff = test.cutoff.iloc[0]
    print(train.groupby(train.cutoff.dt.date).churned.agg(customers="size", churn_rate="mean").round(3))
    print(f"test {test_cutoff.date()}: {len(test):,} customers, churn rate {test.churned.mean():.3f}")

    # Cross-validate on train. The same customer appears in several snapshots, so folds split by customer.
    X, y, groups = train[FEATURES], train.churned, train.index
    cv_rows = []
    for name, model in models().items():
        oof = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
        t = best_f1_threshold(y, oof)
        cv_rows.append({"model": name, "threshold": t, **scores(y, oof, t)})
    cv = pd.DataFrame(cv_rows).set_index("model").round(3)
    chosen = cv.roc_auc.idxmax()
    print("\nCross-validation on train snapshots\n", cv.to_string())

    # Out-of-time test. Baselines: always predict the majority class, and the rule ops would use
    # without a model ("no order in the last 90 days", ranked by recency).
    majority = int(y.mean() >= 0.5)
    test_rows = [
        {"model": f"baseline: always {'churn' if majority else 'stay'}",
         **scores(test.churned, np.full(len(test), majority), 0.5), "roc_auc": 0.5},
        {"model": "baseline: no order in 90 days", **scores(test.churned, test.recency, HORIZON.days)},
    ]
    fitted, proba = {}, {}
    for name, model in models().items():
        fitted[name] = model.fit(X, y)
        proba[name] = model.predict_proba(test[FEATURES])[:, 1]
        test_rows.append({"model": name, **scores(test.churned, proba[name], cv.loc[name, "threshold"])})
    test_res = pd.DataFrame(test_rows).set_index("model").round(3)
    print(f"\nOut-of-time test, chosen model: {chosen}\n", test_res.to_string())

    model = fitted[chosen]
    imp = permutation_importance(model, test[FEATURES], test.churned, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, FEATURES).sort_values()

    # RFM segments, fitted on the test snapshot so each segment's actual churn is known.
    seg = segmenter().fit(test[RFM])
    Z = seg[:-1].transform(test[RFM])
    sil = {k: round(silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)), 3) for k in range(2, 9)}
    names = name_segments(seg, test)
    test["segment"] = pd.Series(seg.predict(test[RFM]), test.index).map(names)
    test["p_churn"] = proba[chosen]
    segments = test.groupby("segment").agg(
        customers=("churned", "size"), median_recency=("recency", "median"), median_orders=("orders_180", "median"),
        median_spend=("spend_180", "median"), spend=("spend_180", "sum"),
        actual_churn=("churned", "mean"), predicted_churn=("p_churn", "mean"))
    segments["spend_share"] = segments.pop("spend") / test.spend_180.sum()
    segments = segments.sort_values("median_spend", ascending=False).round(3)
    print("\nSilhouette by k:", sil)
    print("\nSegments (test snapshot)\n", segments.to_string())

    out, models_dir = ROOT / "results", ROOT / "models"
    out.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    cv.to_csv(out / "cv_results.csv")
    test_res.to_csv(out / "test_results.csv")
    segments.to_csv(out / "segments.csv")
    importance.to_csv(out / "feature_importance.csv", header=["roc_auc_drop"])
    # Refit on all labeled snapshots for end-of-dataset scoring; keep the CV threshold.
    full = pd.concat([train, test])
    final = models()[chosen].fit(full[FEATURES], full.churned)
    joblib.dump({"model": final, "name": chosen, "threshold": cv.loc[chosen, "threshold"],
                 "segmenter": seg, "segment_names": names}, models_dir / "churn.joblib")

    labels = {"logreg": "Logistic regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
    feature_labels = dict(zip(FEATURES, ["Days since last order", "Customer tenure", "Orders, 90 days",
        "Orders, 180 days", "Net spend, 180 days", "Average order value", "Product range",
        "Average order gap", "Return rate", "UK customer", "Peak season"]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for name, p in proba.items():
        fpr, tpr, _ = roc_curve(test.churned, p)
        ax[0].plot(fpr, tpr, label=f"{labels[name]} ({roc_auc_score(test.churned, p):.3f})")
    fpr, tpr, _ = roc_curve(test.churned, test.recency)
    ax[0].plot(fpr, tpr, "k--", label=f"recency rule (AUC {roc_auc_score(test.churned, test.recency):.2f})")
    ax[0].set(title="Can the model rank future churners?", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend()
    importance.rename(index=feature_labels).plot.barh(ax=ax[1], color="#315e78", title=f"What {labels[chosen]} uses to rank customers")
    ax[1].set_xlabel("Drop in test ROC-AUC when shuffled")
    (segments[["actual_churn", "predicted_churn"]] * 100).rename(
        columns={"actual_churn": "Observed", "predicted_churn": "Mean predicted"}
    ).plot.barh(ax=ax[2], color=["#315e78", "#c48b4a"], title="Churn by customer segment")
    ax[2].set(xlabel="Customers who stop buying (%)", ylabel="")
    fig.suptitle(f"90-day churn · test cutoff {test_cutoff:%d %b %Y} · {len(test):,} customers", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "churn.png", dpi=120)


if __name__ == "__main__":
    main()
