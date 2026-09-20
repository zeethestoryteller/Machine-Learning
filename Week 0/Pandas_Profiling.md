# Pandas Profiling


It make a web page that will show:

  Overview of Data
  Univariate Alnalysis
  Bivariate Analysis
  Multivariate Analysis


```python
import pandas as pd

df = pd.read_csv('train.csv')
```


```python
!pip install pandas-profiling

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```

```python
from ydata_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```
