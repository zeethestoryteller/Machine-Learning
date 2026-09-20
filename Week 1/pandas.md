**Pandas Quick Notes**

**Importing & Data Structures**

* `import pandas as pd`: Standard convention for importing the library.
* `pd.Series(data)`: Creates a one-dimensional labeled array capable of holding any data type.
* `pd.DataFrame(data)`: Creates a two-dimensional labeled data structure with columns of potentially different types.

**Reading & Writing Data**

* `pd.read_csv('file.csv')`: Loads data from a comma-separated values file into a DataFrame.
* `df.to_csv('file.csv', index=False)`: Exports a DataFrame to a CSV file without writing row indices.
* `pd.read_excel('file.xlsx')`: Loads data from an Excel file.
* `df.to_excel('file.xlsx')`: Exports a DataFrame to an Excel spreadsheet.

**Data Inspection & Exploration**

* `df.head(n)`: Returns the first $n$ rows of the DataFrame (default is 5).
* `df.tail(n)`: Returns the last $n$ rows of the DataFrame.
* `df.info()`: Displays a concise summary including column data types and non-null counts.
* `df.describe()`: Generates descriptive statistics for numerical columns (mean, std, min, max, percentiles).
* `df.shape`: Returns a tuple representing the dimensionality (rows, columns) of the DataFrame.
* `df.columns`: Returns the column labels of the DataFrame.

**Data Selection & Indexing**

* `df['col_name']`: Selects a single column as a Pandas Series.
* `df[['col1', 'col2']]`: Selects multiple columns as a new DataFrame.
* `df.loc[row_label, col_label]`: Accesses a group of rows and columns by label or boolean array.
* `df.iloc[row_idx, col_idx]`: Accesses rows and columns by integer position (zero-based index).

**Data Cleaning & Manipulation**

* `df.dropna()`: Removes rows or columns that contain missing values (`NaN`).
* `df.fillna(value)`: Replaces missing values with a specified scalar or dictionary.
* `df.duplicated()`: Returns a boolean Series indicating duplicate rows.
* `df.drop_duplicates()`: Returns a DataFrame with duplicate rows removed.
* `df.rename(columns={'old': 'new'})`: Changes column or index labels.
* `df['col'].astype(dtype)`: Converts a column's data type to a specified type.

**Filtering & Conditional Selection**

* `df[df['col'] > value]`: Filters rows based on a specific numerical or boolean condition.
* `df[df['col'].isin([val1, val2])]`: Filters rows where column values match items in a list.

**Grouping & Aggregation**

* `df.groupby('col')`: Splits the data into groups based on criteria specified in a column.
* `df.groupby('col').mean()`: Computes the mean of numerical columns grouped by a categorical column.
* `df.agg(['sum', 'mean'])`: Applies one or more aggregation operations across specified axes.

**Merging & Joining**

* `pd.merge(df1, df2, on='key', how='inner')`: Combines two DataFrames using database-style join operations.
* `pd.concat([df1, df2], axis=0)`: Concatenates pandas objects along a particular axis (stacking rows or columns).
