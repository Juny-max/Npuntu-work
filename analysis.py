"""Npontu Technologies e-commerce customer churn analysis.

This program generates synthetic customer data, cleans it, engineers useful
features, explores customer behaviour, uses Dask for partitioned processing,
trains a churn prediction model, evaluates it, and saves charts/results.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
REFERENCE_DATE = pd.Timestamp("2026-07-16")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

for folder in (DATA_DIR, OUTPUT_DIR, CHART_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n_customers: int = 12000) -> pd.DataFrame:
    """Generate realistic customer-level e-commerce activity.

    A few missing values, invalid entries, and duplicates are deliberately
    inserted so the cleaning stage demonstrates real data-quality work.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    age = np.clip(rng.normal(34, 11, n_customers).round(), 18, 75)
    gender = rng.choice(["Female", "Male", "Other"], n_customers, p=[0.49, 0.49, 0.02])
    city = rng.choice(
        ["Accra", "Kumasi", "Tema", "Takoradi", "Cape Coast", "Tamale"],
        n_customers,
        p=[0.42, 0.22, 0.13, 0.09, 0.07, 0.07],
    )
    device = rng.choice(["Mobile", "Desktop", "Tablet"], n_customers, p=[0.68, 0.25, 0.07])
    membership = rng.choice(["Basic", "Silver", "Gold"], n_customers, p=[0.62, 0.27, 0.11])

    tenure_days = rng.integers(30, 1460, n_customers)
    recency_days = np.clip(rng.exponential(52, n_customers).astype(int), 0, 365)
    signup_date = REFERENCE_DATE - pd.to_timedelta(tenure_days, unit="D")
    last_purchase_date = REFERENCE_DATE - pd.to_timedelta(recency_days, unit="D")

    website_visits = np.maximum(1, rng.poisson(15, n_customers))
    pages_viewed = website_visits * rng.integers(2, 8, n_customers)
    avg_session_minutes = np.clip(rng.gamma(3.0, 3.0, n_customers), 0.5, 45).round(2)
    purchases = np.minimum(website_visits, rng.poisson(6, n_customers))
    average_basket = np.clip(rng.lognormal(4.15, 0.55, n_customers), 10, 900)
    total_spent = (purchases * average_basket).round(2)
    cart_additions = purchases + rng.poisson(4, n_customers)
    abandoned_carts = np.maximum(0, cart_additions - purchases)
    support_tickets = rng.poisson(1.0, n_customers)
    email_open_rate = np.clip(rng.beta(2.5, 2.2, n_customers), 0, 1).round(3)
    discount_usage = np.clip(rng.beta(2.0, 3.5, n_customers), 0, 1).round(3)

    # Churn is more likely after long inactivity, many abandoned carts, and
    # support problems. Purchases, engagement, and premium membership reduce it.
    premium = np.isin(membership, ["Silver", "Gold"]).astype(int)
    churn_logit = (
        -1.0
        + 0.025 * (recency_days - 45)
        + 0.14 * abandoned_carts
        + 0.18 * support_tickets
        - 0.08 * purchases
        - 0.95 * email_open_rate
        - 0.35 * premium
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n_customers + 1)],
            "age": age,
            "gender": gender,
            "city": city,
            "device": device,
            "membership": membership,
            "signup_date": signup_date,
            "last_purchase_date": last_purchase_date,
            "website_visits": website_visits,
            "pages_viewed": pages_viewed,
            "avg_session_minutes": avg_session_minutes,
            "purchases": purchases,
            "total_spent": total_spent,
            "cart_additions": cart_additions,
            "abandoned_carts": abandoned_carts,
            "support_tickets": support_tickets,
            "email_open_rate": email_open_rate,
            "discount_usage": discount_usage,
            "churn": churn,
        }
    )

    # Insert realistic data-quality problems for the cleaning demonstration.
    for column in ["age", "avg_session_minutes", "email_open_rate"]:
        missing_rows = rng.choice(df.index, size=int(0.01 * n_customers), replace=False)
        df.loc[missing_rows, column] = np.nan

    invalid_age_rows = rng.choice(df.index, size=30, replace=False)
    invalid_spend_rows = rng.choice(df.index, size=30, replace=False)
    df.loc[invalid_age_rows, "age"] = 150
    df.loc[invalid_spend_rows, "total_spent"] = -100

    duplicate_rows = df.sample(60, random_state=RANDOM_STATE)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    df.to_csv(DATA_DIR / "synthetic_customer_data_raw.csv", index=False)
    return df


def clean_and_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean erroneous records and create modelling features."""
    quality_before = {
        "rows": int(len(df)),
        "duplicate_customer_ids": int(df.duplicated(subset="customer_id").sum()),
        "missing_values": int(df.isna().sum().sum()),
        "invalid_ages": int(((df["age"] < 18) | (df["age"] > 90)).sum()),
        "negative_spend": int((df["total_spent"] < 0).sum()),
    }

    clean = df.drop_duplicates(subset="customer_id").copy()
    clean.loc[~clean["age"].between(18, 90), "age"] = np.nan
    clean.loc[clean["total_spent"] < 0, "total_spent"] = np.nan

    numeric_columns = clean.select_dtypes(include="number").columns
    for column in numeric_columns:
        clean[column] = clean[column].fillna(clean[column].median())

    categorical_columns = clean.select_dtypes(include="object").columns
    for column in categorical_columns:
        clean[column] = clean[column].fillna(clean[column].mode()[0])

    clean["signup_date"] = pd.to_datetime(clean["signup_date"])
    clean["last_purchase_date"] = pd.to_datetime(clean["last_purchase_date"])

    # RFM-style and engagement features used by the model.
    clean["tenure_days"] = (REFERENCE_DATE - clean["signup_date"]).dt.days
    clean["recency_days"] = (REFERENCE_DATE - clean["last_purchase_date"]).dt.days
    clean["average_order_value"] = clean["total_spent"] / clean["purchases"].replace(0, 1)
    clean["cart_abandonment_rate"] = clean["abandoned_carts"] / clean["cart_additions"].replace(0, 1)
    clean["purchase_frequency"] = clean["purchases"] / (clean["tenure_days"] / 30).clip(lower=1)
    clean["engagement_score"] = (
        0.35 * clean["website_visits"]
        + 0.15 * clean["pages_viewed"]
        + 8.0 * clean["email_open_rate"]
        + 0.25 * clean["avg_session_minutes"]
    )
    clean["age_group"] = pd.cut(
        clean["age"],
        bins=[17, 24, 34, 44, 54, 90],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    ).astype(str)

    quality_after = {
        "rows": int(len(clean)),
        "duplicate_customer_ids": int(clean.duplicated(subset="customer_id").sum()),
        "missing_values": int(clean.isna().sum().sum()),
        "invalid_ages": int(((clean["age"] < 18) | (clean["age"] > 90)).sum()),
        "negative_spend": int((clean["total_spent"] < 0).sum()),
    }
    quality_report = {"before_cleaning": quality_before, "after_cleaning": quality_after}

    clean.to_csv(DATA_DIR / "customer_data_cleaned.csv", index=False)
    with open(OUTPUT_DIR / "data_quality_report.json", "w", encoding="utf-8") as file:
        json.dump(quality_report, file, indent=2)
    return clean, quality_report


def perform_eda(df: pd.DataFrame) -> dict:
    """Produce summary tables and readable business visualisations."""
    sns.set_theme(style="whitegrid", palette="deep")

    insights = {
        "customers": int(len(df)),
        "churn_rate_percent": round(float(df["churn"].mean() * 100), 2),
        "average_customer_spend": round(float(df["total_spent"].mean()), 2),
        "average_purchases": round(float(df["purchases"].mean()), 2),
        "highest_churn_membership": str(df.groupby("membership")["churn"].mean().idxmax()),
        "highest_churn_device": str(df.groupby("device")["churn"].mean().idxmax()),
    }

    membership_summary = (
        df.groupby("membership", observed=False)
        .agg(customers=("customer_id", "count"), churn_rate=("churn", "mean"), average_spend=("total_spent", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    membership_summary["churn_rate"] *= 100
    membership_summary.round(2).to_csv(OUTPUT_DIR / "membership_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.countplot(data=df, x="churn", ax=axes[0, 0], color="#2F80ED")
    axes[0, 0].set_title("Customer Churn Distribution")
    axes[0, 0].set_xticklabels(["Active", "Churned"])

    churn_membership = df.groupby("membership", observed=False)["churn"].mean().mul(100).reset_index()
    sns.barplot(data=churn_membership, x="membership", y="churn", ax=axes[0, 1])
    axes[0, 1].set_title("Churn Rate by Membership")
    axes[0, 1].set_ylabel("Churn rate (%)")

    sns.boxplot(data=df, x="churn", y="recency_days", ax=axes[1, 0], color="#56CCF2")
    axes[1, 0].set_title("Days Since Last Purchase vs Churn")
    axes[1, 0].set_xticklabels(["Active", "Churned"])

    city_spend = df.groupby("city", observed=False)["total_spent"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=city_spend, y="city", x="total_spent", ax=axes[1, 1], color="#27AE60")
    axes[1, 1].set_title("Average Spend by City")
    axes[1, 1].set_xlabel("Average spend (GHS)")

    plt.tight_layout()
    fig.savefig(CHART_DIR / "customer_insights.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    correlation_columns = [
        "churn", "recency_days", "purchases", "total_spent", "abandoned_carts",
        "support_tickets", "email_open_rate", "engagement_score",
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df[correlation_columns].corr(), annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax)
    ax.set_title("Correlation Between Customer Behaviour Features")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with open(OUTPUT_DIR / "eda_insights.json", "w", encoding="utf-8") as file:
        json.dump(insights, file, indent=2)
    return insights


def run_dask_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Use Dask partitions to simulate scalable/distributed data processing."""
    import dask.dataframe as dd

    dask_df = dd.from_pandas(df, npartitions=8)
    result = (
        dask_df.groupby("city")
        .agg({"customer_id": "count", "total_spent": "mean", "churn": "mean"})
        .compute()
        .rename(
            columns={
                "customer_id": "customers",
                "total_spent": "average_spend",
                "churn": "churn_rate",
            }
        )
    )
    result["churn_rate"] *= 100
    result = result.sort_values("average_spend", ascending=False).round(2)
    result.to_csv(OUTPUT_DIR / "dask_city_analysis.csv")
    return result


def train_churn_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Train, cross-validate, evaluate, and save a churn classifier."""
    target = "churn"
    excluded = ["customer_id", "signup_date", "last_purchase_date", target]
    features = [column for column in df.columns if column not in excluded]
    X = df[features]
    y = df[target]

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1_scores = cross_val_score(pipeline, X, y, cv=cross_validation, scoring="f1", n_jobs=-1)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "cross_validation_f1_mean": round(float(cv_f1_scores.mean()), 4),
        "cross_validation_f1_std": round(float(cv_f1_scores.std()), 4),
        "test_customers": int(len(y_test)),
    }

    report = classification_report(y_test, predictions, target_names=["Active", "Churned"])
    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions),
        display_labels=["Active", "Churned"],
    ).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion Matrix")
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    axes[1].set_title("ROC Curve")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "model_evaluation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    top_features = importance_df.head(12).sort_values("importance")
    ax.barh(top_features["feature"].str.replace("numeric__", "").str.replace("categorical__", ""), top_features["importance"], color="#2F80ED")
    ax.set_title("Top Drivers of Customer Churn")
    ax.set_xlabel("Feature importance")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(pipeline, OUTPUT_DIR / "churn_model.joblib")
    return pipeline, metrics, importance_df


def create_business_recommendations(df: pd.DataFrame, importance_df: pd.DataFrame) -> list[str]:
    """Turn findings into actions the e-commerce business can take."""
    high_risk_recency = int(df.groupby("churn")["recency_days"].mean().loc[1])
    churn_by_membership = df.groupby("membership")["churn"].mean().sort_values(ascending=False)
    highest_risk_plan = str(churn_by_membership.index[0])
    top_driver = importance_df.iloc[0]["feature"].replace("numeric__", "").replace("categorical__", "")

    recommendations = [
        f"Launch automated re-engagement messages before customers reach about {high_risk_recency} days without purchasing.",
        f"Prioritize retention offers for {highest_risk_plan} members, the membership group with the highest observed churn rate.",
        "Send cart reminders and limited-time incentives to customers with high cart-abandonment rates.",
        "Route customers with repeated support tickets to proactive service follow-ups before dissatisfaction causes churn.",
        "Create a CRM risk segment from the model probability and contact the highest-risk customers first.",
        f"Monitor {top_driver} closely because the trained model identified it as the strongest churn driver.",
    ]
    with open(OUTPUT_DIR / "business_recommendations.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(f"{index}. {text}" for index, text in enumerate(recommendations, 1)))
    return recommendations


def main() -> None:
    """Run the complete assignment from start to finish."""
    print("1/6 Generating synthetic customer data...")
    raw_df = generate_synthetic_data()
    print(f"Generated {len(raw_df):,} rows.")

    print("2/6 Cleaning data and engineering features...")
    clean_df, quality_report = clean_and_engineer(raw_df)
    print(json.dumps(quality_report, indent=2))

    print("3/6 Performing exploratory analysis and creating charts...")
    eda_insights = perform_eda(clean_df)
    print(json.dumps(eda_insights, indent=2))

    print("4/6 Running partitioned analysis with Dask...")
    dask_result = run_dask_analysis(clean_df)
    print(dask_result)

    print("5/6 Training and validating churn model...")
    _, metrics, importance_df = train_churn_model(clean_df)
    print(json.dumps(metrics, indent=2))

    print("6/6 Creating business recommendations...")
    recommendations = create_business_recommendations(clean_df, importance_df)
    for index, recommendation in enumerate(recommendations, 1):
        print(f"{index}. {recommendation}")

    print(f"\nComplete. Results are saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
