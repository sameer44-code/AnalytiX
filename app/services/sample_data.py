import pandas as pd
import numpy as np
import random

def generate_retail_sales_dataset() -> pd.DataFrame:
    """Generates an enterprise retail sales dataset with realistic dirty patterns for testing."""
    np.random.seed(42)
    random.seed(42)
    n = 600

    categories = ["Enterprise Hardware", "Cloud Software", "Office Infrastructure", "Security Appliances", "Consulting Services"]
    regions = ["North America", "EMEA", "APAC", "LATAM", None]
    channels = ["Direct Sales", "Partner Network", "Online Portal", "Reseller"]

    data = {
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n)],
        "Product_Category": np.random.choice(categories, n, p=[0.25, 0.30, 0.20, 0.15, 0.10]),
        "Region": np.random.choice(regions, n, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
        "Sales_Channel": np.random.choice(channels, n),
        "Unit_Price": np.round(np.random.gamma(shape=5, scale=50, size=n), 2),
        "Quantity": np.random.choice([1, 2, 3, 4, 5, 8, 10, 15, 20], n),
        "Discount_Pct": np.round(np.random.beta(a=1, b=5, size=n) * 100, 1),
        "Shipping_Days": np.random.choice([1, 2, 3, 5, 7, 10, 14, None], n, p=[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]),
        "Customer_Rating": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, None], n, p=[0.05, 0.08, 0.22, 0.40, 0.20, 0.05]),
        "Returned": np.random.choice(["No", "Yes", None], n, p=[0.82, 0.15, 0.03])
    }

    df = pd.DataFrame(data)
    df["Revenue"] = np.round(df["Unit_Price"] * df["Quantity"] * (1 - df["Discount_Pct"] / 100.0), 2)
    df["Profit_Margin_Pct"] = np.round(np.random.normal(loc=28.5, scale=12.0, size=n), 1)

    # Inject realistic dirty data patterns:
    # 1. Missing values in Unit_Price
    df.loc[np.random.choice(n, 28, replace=False), "Unit_Price"] = np.nan
    # 2. Inconsistent negative prices or quantities (unusual values)
    df.loc[12, "Unit_Price"] = -45.0
    df.loc[45, "Quantity"] = 0
    # 3. Extreme outliers in Revenue and Quantity
    df.loc[88, "Quantity"] = 350  # Huge bulk order
    df.loc[142, "Revenue"] = 18500.0  # Massive outlier
    # 4. Duplicate rows
    duplicates = df.iloc[[10, 25, 40, 75, 110]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


def generate_saas_churn_dataset() -> pd.DataFrame:
    """Generates an enterprise B2B SaaS dataset."""
    np.random.seed(101)
    random.seed(101)
    n = 500

    tiers = ["Starter", "Professional", "Enterprise", "Custom"]
    industries = ["Fintech", "Healthcare", "E-commerce", "Manufacturing", "EdTech", None]

    data = {
        "Account_ID": [f"ACCT-{2000 + i}" for i in range(n)],
        "Subscription_Tier": np.random.choice(tiers, n, p=[0.35, 0.40, 0.20, 0.05]),
        "Industry": np.random.choice(industries, n, p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05]),
        "Annual_Recurring_Revenue": np.round(np.random.lognormal(mean=9.5, sigma=0.8, size=n), 0),
        "Monthly_Active_Users": np.random.choice(range(5, 500), n),
        "Support_Tickets_Logged": np.random.choice(range(0, 25), n),
        "NPS_Score": np.random.choice([-10, 0, 5, 7, 8, 9, 10, None], n, p=[0.05, 0.05, 0.1, 0.2, 0.3, 0.15, 0.1, 0.05]),
        "Contract_Length_Months": np.random.choice([1, 12, 24, 36], n, p=[0.2, 0.5, 0.2, 0.1]),
        "Feature_Adoption_Rate": np.round(np.random.uniform(0.15, 0.98, n), 2),
        "Churned": np.random.choice(["No", "Yes"], n, p=[0.78, 0.22])
    }

    df = pd.DataFrame(data)
    # Dirty patterns
    df.loc[np.random.choice(n, 20, replace=False), "Annual_Recurring_Revenue"] = np.nan
    df.loc[15, "Annual_Recurring_Revenue"] = 450000.0  # Whale outlier
    # Duplicates
    df = pd.concat([df, df.iloc[[5, 18, 99]]], ignore_index=True)
    return df


def generate_financial_risk_dataset() -> pd.DataFrame:
    """Generates an enterprise financial credit and portfolio risk dataset."""
    np.random.seed(202)
    random.seed(202)
    n = 550

    loan_purposes = ["Debt Consolidation", "Business Expansion", "Working Capital", "Equipment", "Real Estate"]
    credit_grades = ["A", "B", "C", "D", "E", None]

    data = {
        "Applicant_ID": [f"APP-{5000 + i}" for i in range(n)],
        "Credit_Score": np.random.choice(range(480, 850), n),
        "Annual_Income": np.round(np.random.lognormal(mean=11.2, sigma=0.6, size=n), 0),
        "Loan_Amount": np.round(np.random.uniform(5000, 150000, n), -2),
        "Debt_To_Income_Ratio": np.round(np.random.uniform(0.05, 0.65, n), 3),
        "Interest_Rate_Pct": np.round(np.random.uniform(3.5, 24.5, n), 2),
        "Loan_Purpose": np.random.choice(loan_purposes, n),
        "Credit_Grade": np.random.choice(credit_grades, n, p=[0.25, 0.30, 0.25, 0.10, 0.05, 0.05]),
        "Delinquent_Months_2Yrs": np.random.choice([0, 1, 2, 3, 5, None], n, p=[0.75, 0.12, 0.06, 0.04, 0.01, 0.02]),
        "Default_Status": np.random.choice(["Performing", "Defaulted"], n, p=[0.88, 0.12])
    }

    df = pd.DataFrame(data)
    df.loc[np.random.choice(n, 22, replace=False), "Annual_Income"] = np.nan
    df.loc[20, "Debt_To_Income_Ratio"] = 2.85  # Outlier DTI
    df = pd.concat([df, df.iloc[[12, 34]]], ignore_index=True)
    return df

SAMPLE_CATALOG = {
    "retail_sales": {
        "name": "Global Enterprise Retail & Marketing Transactions",
        "file_type": "csv",
        "description": "Enterprise multi-channel transactions with missing unit prices, return rates, regional variance, and shipping days.",
        "generator": generate_retail_sales_dataset
    },
    "saas_churn": {
        "name": "B2B SaaS Customer Churn & Revenue Analytics",
        "file_type": "csv",
        "description": "Recurring revenue, feature adoption metrics, support ticket volume, NPS scores, and subscription churn markers.",
        "generator": generate_saas_churn_dataset
    },
    "financial_risk": {
        "name": "Commercial Credit Risk & Portfolio Delinquency",
        "file_type": "csv",
        "description": "Applicant credit scores, debt-to-income ratios, loan amounts, credit grades, and default outcomes.",
        "generator": generate_financial_risk_dataset
    }
}
