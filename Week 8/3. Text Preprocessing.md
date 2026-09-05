# Load Text data with csv file


Downloading SMS Collection Data 
```python
!wget https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
```

unziping File
```python
!unzip sms+spam+collection.zip
```

Organising Data
```python
sms_data = pd.read_csv('/content/SMSSpamCollection',
                       sep='\t',
                       header= None,
                       names= ['label', 'text'])
```
