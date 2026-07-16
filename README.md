# E-commerce Customer Behaviour and Churn Analysis

Applicant: Fianko Junior Owusu  
Role: Intelligent Systems Services Engineer  
Assignment: Analyzing Customer Behavior for E-commerce Insights

## Project overview

This project creates a synthetic e-commerce customer dataset and uses it to:

1. Demonstrate data cleaning and exploratory data analysis.
2. Engineer recency, frequency, monetary, cart, and engagement features.
3. Predict customer churn using a Random Forest classifier.
4. Validate performance using a held-out test set and five-fold cross-validation.
5. Use Dask partitions to demonstrate scalable data processing.
6. Turn the findings into business recommendations.

## How to run it

Open a terminal inside this project folder and run:

```bash
python -m venv .venv
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
pip install -r requirements.txt
```

You can now either open `ecommerce_customer_analysis.ipynb` in VS Code and click **Run All**, or run the complete program directly:

```bash
python analysis.py
```

## Output files

Running the analysis automatically creates:

- Raw and cleaned CSV datasets in `data/`.
- Data-quality and exploratory-analysis summaries in `outputs/`.
- Model metrics and a classification report.
- A saved churn prediction model.
- Customer-insight, correlation, model-evaluation, and feature-importance charts.
- Actionable business recommendations.

## Big data technology choice

Dask was selected because it provides a pandas-like API while dividing a dataset into partitions that may be processed in parallel. This notebook uses eight partitions to simulate a scalable workflow. The same code can later process data that is larger than a computer's memory or run on a Dask cluster without replacing the complete analysis pipeline.

## Important note

The dataset is synthetic and is intended to demonstrate the methodology. Before using this model in production, it should be retrained and monitored using consented, properly governed real customer data.
