
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')


def eda_by_ai(df):
  # 1. SETUP & DATA INFO
  print('--- DATASET INFO ---')
  print(df.info())
  print('\n--- MISSING VALUES ---')
  print(df.isnull().sum().head(3))
  print('\n--- DUPLICATED ROWS ---', df.duplicated().sum())

  # 2. STATISTICAL DESCRIPTIONS & CORRELATION
  print('\n--- NUMERICAL DESCRIPTION ---')
  print(df.describe().head(3))

  print('\n--- OBJECT/CATEGORICAL DESCRIPTION ---')
  print(df.describe(include=['O']).head(3))

  numeric_df = df.select_dtypes(include=[np.number])
  if not numeric_df.empty:
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5
    )
    plt.title('Correlation Matrix', fontsize=12)
    plt.tight_layout()
    plt.show()

  # 3. UNIVARIATE ANALYSIS (Max 2 plots)
  num_cols = numeric_df.columns.tolist()
  cat_cols = df.select_dtypes(include=['O', 'category']).columns.tolist()

  if len(num_cols) > 0:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[num_cols[0]], kde=True, color='royalblue')
    plt.title(f'Distribution of {num_cols[0]}', fontsize=10)
    plt.tight_layout()
    plt.show()

  if len(cat_cols) > 0:
    plt.figure(figsize=(6, 3))
    sns.countplot(
        data=df, x=cat_cols[0], order=df[cat_cols[0]].value_counts().index[:3]
    )
    plt.title(f'Frequency Count of {cat_cols[0]}', fontsize=10)
    plt.tight_layout()
    plt.show()

  # 4. BIVARIATE ANALYSIS (Max 2 plots)
  if len(cat_cols) > 0 and len(num_cols) > 0:
    plt.figure(figsize=(6, 3))
    sns.barplot(data=df, x=cat_cols[0], y=num_cols[0], ci=None)
    plt.title(f'{num_cols[0]} by {cat_cols[0]}', fontsize=10)
    plt.tight_layout()
    plt.show()

  if len(num_cols) >= 2:
    plt.figure(figsize=(6, 3))
    sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1])
    plt.title(f'{num_cols[1]} vs {num_cols[0]}', fontsize=10)
    plt.tight_layout()
    plt.show()

  # 5. TIME SERIES ANALYSIS (Max 2 plots)
  date_cols = df.select_dtypes(
      include=['datetime64[ns]', 'datetime']
  ).columns.tolist()
  if not date_cols:
    for col in df.columns:
      if 'date' in col.lower():
        try:
          df[col] = pd.to_datetime(df[col])
          date_cols.append(col)
          break
        except Exception:
          pass

  if date_cols and len(num_cols) > 0:
    d_col = date_cols[0]
    n_col = num_cols[0]
    ts_df = df.set_index(d_col).resample('M')[n_col].sum().reset_index()

    plt.figure(figsize=(6, 3))
    sns.lineplot(data=ts_df, x=d_col, y=n_col, marker='o')
    plt.title(f'Monthly {n_col} Trend Over Time', fontsize=10)
    plt.tight_layout()
    plt.show()

  # 6. MULTIVARIATE ANALYSIS (Max 2 plots)
  if len(num_cols) >= 2 and len(cat_cols) > 0:
    plt.figure(figsize=(6, 3))
    sns.scatterplot(
        data=df, x=num_cols[0], y=num_cols[1], hue=cat_cols[0], alpha=0.7
    )
    plt.title(f'Multivariate: {num_cols[1]} vs {num_cols[0]}', fontsize=10)
    plt.tight_layout()
    plt.show()
