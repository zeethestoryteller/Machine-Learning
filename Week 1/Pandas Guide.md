# Pandas Guide

A structured reference built from your notebook, organized around the CRUD framework it uses: **Create, Read, Update, Delete**.

---

## 1. Setup

```python
import pandas as pd
import numpy as np
```

---

## 2. Create: Series & DataFrames

### Series (one column of data)

```python
s1 = pd.Series([1, 2, 3, 4, 5])
s1.index      # default: RangeIndex(0..4)
s1.values     # underlying numpy array
```

You can give a Series a custom index, e.g. a date range:

```python
s2 = pd.Series([1, 2, 3, 4, 5, 6],
                index=pd.date_range("2023/09/25", periods=6))
```

### DataFrame (a table)

Built from a dictionary of equal-length lists — each key becomes a column:

```python
d1 = {"name": ["mayur", "nitin", "deepak", "prashant", "rishabh"],
      "roll": [1, 2, 3, 4, 5]}
df1 = pd.DataFrame(d1)
```

Dict methods still work on the raw dict before conversion: `d1.keys()`, `d1.values()`, `d1.items()`.

### Combining DataFrames

- **`pd.concat([df1, df2], axis=1)`** — stick DataFrames side by side (axis=1) or stack rows (axis=0). If row counts don't match, mismatched rows fill with `NaN`.
- **`df1.set_index("roll")`** — make a column the index (useful before joining).
- **`df1.join(df2.set_index("roll"))`** — merge on the index, keeping `df1`'s rows.

### Reading files

```python
df = pd.read_csv("path/to/file.csv", usecols=["Car_Name", "Year", "Selling_Price", "Transmission"])
```

- `usecols` loads only the columns you need (faster, less memory).
- Pandas can also read directly from a shareable URL (e.g. a reformatted Google Drive download link).
- `pd.read_json()` works the same way for JSON sources.

---

## 3. Read: Selecting, Filtering, Sorting

### Column access & new columns

```python
df["Car_Name"]                                  # select a column
df.columns                                       # list all column names
df["present-sell"] = df["Present_Price"] - df["Selling_Price"]   # create a new column
```

### `loc` vs `iloc`

| | Selects by | Slicing |
|---|---|---|
| `df.iloc[r, c]` | integer position (starts at 0) | end index **excluded**, like Python lists |
| `df.loc[r, c]` | labels (index/column names) | end label **included** |

```python
df.iloc[0, 0]                     # first row, first column
df.iloc[::2, :]                   # every other row, all columns
df.iloc[1:60:2, 0:8:2]            # start:stop:step on both axes
df.iloc[[0, 5, 3], [3, 7, 4]]     # arbitrary row/column positions
df.loc[:, "Present_Price":"Owner"]  # label-based slice, inclusive of "Owner"
```

Assigning a value works the same way: `df.iloc[1, 1] = 2015`.

### Boolean indexing (filtering by condition)

```python
df[(df["Year"] == 2014) & (df["Present_Price"] > 3.3)]                 # AND
df[(df["Year"] == 2014) | (df["Fuel_Type"] == "Diesel")]               # OR
df[df["Fuel_Type"].isin(["Diesel", "Petrol"])]                         # value in a list
df[~df["Car_Name"].isin(["ciaz", "sx4"])]                              # NOT (negation with ~)
```

Use `&`, `|`, `~` (not `and`/`or`/`not`) and wrap each condition in parentheses.

### Sorting

```python
df.sort_index()                                        # sort by index
df.sort_values(by="Selling_Price", ascending=False)     # sort by column value
test.reset_index()                                      # rebuild index as 0,1,2,...
```

### Quick inspection

```python
df.head()      # first 5 rows
df.tail()      # last 5 rows
df.shape       # (rows, columns)
df.dtypes      # column data types
df["Year"] = df["Year"].astype("int32")   # change a column's type
```

---

## 4. Update: Handling Missing & Messy Data

### Finding nulls

```python
df.isnull().sum()                 # null count per column
hd[hd.isnull().sum(axis=1) > 3]   # rows with more than 3 nulls
df2.isnull().sum(axis=1).max()    # worst-case null count in any single row
```

### Dropping nulls

```python
hd.dropna()                  # drop any row containing a null
hd.dropna(thresh=11)         # keep rows with at least 11 non-null values
hd.dropna(thresh=(16 - 3))   # keep rows with ≤ 3 nulls, when there are 16 columns total
```

`thresh` sets the **minimum number of non-null values a row needs to survive** — so to drop rows with more than *N* nulls out of *C* columns, use `thresh=(C - N)`.

### Dropping columns

```python
hd.drop("fbs", axis=1, inplace=True)   # axis=1 = column; inplace edits df directly
```

### Filling / replacing values

```python
hd.replace(np.nan, 0)                    # replace all nulls with 0
hd2.replace(method="bfill", limit=1)     # backward-fill, at most 1 step
hd2.replace(method="ffill", limit=1)     # forward-fill, at most 1 step
```

(`fillna()` is the more common/direct tool for this — `df["col"].fillna(value)`.)

### Duplicates & category counts

```python
df["Fuel_Type"].value_counts()                        # counts per category
pd.crosstab(df["Fuel_Type"], df["Seller_Type"], margins=True)   # cross-tab of two categorical columns, with totals
df.drop_duplicates()
```

---

## 5. Descriptive Statistics

```python
hd.describe()                       # count, mean, std, min, quartiles, max — numeric columns only
hd.describe().loc["mean", "age"]    # pull one specific stat
hd["age"].mean()
hd["age"].median()
hd["chol"].var()                    # variance
hd["chol"].std()                    # standard deviation
```

### Quantiles & the IQR (outlier detection)

```python
Q1 = hd["age"].quantile(0.25)
Q3 = hd["age"].quantile(0.75)
IQR = Q3 - Q1
```

**Five-number summary:** min, Q1, median, Q3, max — this is what a boxplot visualizes.

**Boxplot whiskers:**
- Upper whisker: `Q3 + 1.5 * IQR`
- Lower whisker: `Q1 - 1.5 * IQR`

Anything beyond the whiskers is a candidate outlier:

```python
df[df["age"] > q3 + 1.5 * iqr]   # rows flagged as outliers
hd["age"].plot(kind="box")       # visualize
```

### Outliers under a normal distribution (empirical rule)

For normally distributed data:
- μ ± 1σ ≈ 68% of data
- μ ± 2σ ≈ 95% of data
- μ ± 3σ ≈ 99.7% of data

Points far outside these bands (commonly beyond 2–3σ) are statistical outliers.

```python
hd["age"].hist()   # visualize the distribution
```

---

## 6. Dates

```python
date = pd.date_range("2023-09-29", periods=6)
date.year
date.month
date.day
```

---

## 7. Grouping & Pivoting (Advanced)

```python
hd.groupby("sex").mean()["chol"]              # average cholesterol per sex
hd.groupby("dataset").count()                 # row counts per group

pd.crosstab(hd["dataset"], hd["cp"], margins=True)   # frequency table

pd.pivot_table(hd, values="dataset",
                index=["sex", "cp"],
                columns=["exang"],
                aggfunc="count",
                fill_value=0)
```

- **`groupby`** splits data into groups and applies an aggregate (`mean`, `count`, `sum`, etc.) to each.
- **`crosstab`** is a quick frequency count between two categorical columns.
- **`pivot_table`** is a more flexible groupby: choose which column's values to aggregate, which columns become row/column labels, and how to aggregate (`aggfunc`).

Related reshaping tools worth knowing (mentioned in your notes but not yet used): `pivot`, `melt`, `reindex`, `merge`.

---

## 8. Train/Test Splitting (via scikit-learn, works on DataFrames)

```python
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
```

This returns two DataFrames (80/20 split by default row sampling) — pandas indexing carries over, so `test.sort_index()` or `test.reset_index()` can clean up the row order afterward.

---

## Quick Reference Table

| Task | Method |
|---|---|
| Filter rows by condition | `df[condition]`, combine with `&` `|` `~` |
| Select by position | `df.iloc[rows, cols]` |
| Select by label | `df.loc[rows, cols]` |
| Check for nulls | `df.isnull().sum()` |
| Drop null rows | `df.dropna()` / `df.dropna(thresh=n)` |
| Fill/replace values | `df.fillna(val)` / `df.replace(old, new)` |
| Value frequency | `df["col"].value_counts()` |
| Cross-tabulate two columns | `pd.crosstab(a, b, margins=True)` |
| Group and aggregate | `df.groupby("col").agg_func()` |
| Reshape/summarize | `pd.pivot_table(...)` |
| Combine DataFrames | `pd.concat()`, `df.join()`, `pd.merge()` |
| Summary stats | `df.describe()`, `.mean()`, `.median()`, `.std()`, `.var()`, `.quantile()` |
| Spot outliers | IQR method or `μ ± kσ` (empirical rule) |

---

*A note on a few things worth double-checking in the original notebook: `inplce=True` (cell 90) is a typo for `inplace=True`, and `hd.replace()` with no arguments (cell 155) won't actually replace nulls — `fillna()` is the more reliable tool for that.*
