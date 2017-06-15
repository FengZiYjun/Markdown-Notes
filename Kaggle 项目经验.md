# Kaggle 项目经验

## 1. read and write
use pandas package
`import pandas as pd`
Provide the path of the file, pd.read_csv() returns a DataFrame object.
`dfObj = pd.read_csv('file_path')`
To write DataFrame into a file, use to_csv method.
parameter: index(whether to display the indexing column), header(assign the first row in a list)
`dfObj.to_csv('file_path',index=false, header=['col1','col2','col3'])`

## 2. basic opearation of DataFrame
### 2.1 Add
- add a column
Assign the new key with a new pandas.Series object to add a new column at the **right** side of the table
`df['Key'] = pd.Series()`
- add a row
`df.loc[index_number] = list()`
`df.ix[index_number] = list()`

### 2.2 delete
- delete a column
`del(df['Key'])`
- delete a row
To use specific feature. Eg.abandon all the row with null in 'column_name' column
`df = df[df.column_name.notnull()]`

### 2.3 visit
`df['column_name']` returns a Series object with this column
`df.ix[index]` returns a ? object with this row
`df.columns` show all the keys/features

## 3. how to deal with NaN
- dispose of row with all NaN
`df.dropna(how='any')`
- drop rows with NaN in specific columns by
`df = df[df.column_name.notnull()]`
- fill the NaN with data
`df.fillna(method='ffill')  #'bfill'`
`df.fillna(0)`

## 4. convert DataFrame into list
```
import numpy as np
l = np.array(df)
l = [list(x) for x in l] # convert array_object to list
l = [[int(x) for x in d] for d in l] # convert numpt_int64 to int
```