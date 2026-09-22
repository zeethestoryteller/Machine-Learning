# Pandas Profiling and Y data Profiling


It make a web page that will show:

* Overview of Data
* Univariate Alnalysis
* Bivariate Analysis
* Multivariate Analysis
#
### Data Loading
```python
import pandas as pd

file_id = "1ZzoufsZLGFGNV_Vk4GDoQk9xVtFIoIqg"
url = f"https://drive.google.com/uc?id={file_id}&export=download"
df = pd.read_csv(url)
```
  
# 
### Pandas Profiling

```python
!pip install pandas-profiling

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```
#
### Y data Profiling
```python
pip install ydata-profiling

from ydata_profiling import ProfileReport
prof = ProfileReport(df, title="Pandas Profiling Report")
prof.to_file(output_file='output.html')
```
