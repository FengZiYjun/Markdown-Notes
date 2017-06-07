# Notes of Learning Data Science

## Pandas: basic application of Series & DataFrame

### 1. preparation 
```    python
import pandas as pd
from pandas import Series,DataFrame
```

### 2. Series
```    python
ser = Series([1,2,3]) # with array
ser = Series([1,2,3,4],index = ['a','b','c','d']) 
ser = Series({'p':1,'a':2,'n':3})  # with  dictionary
ser.values 
ser.index
ser[2] # call by default index
ser['b'] # call by my index(s)
ser[['a','c']] 
pd.isnull(ser)  # check not-a-number
pd.notnull(ser)
ser.isnull()
ser.notnull()
    	
ser1 + ser2 # add value with the same index
ser.name = 'python' # set the name of the series
ser.index.name = 'index name' # set the name of the index columne
ser.index = ['a','b','v'] # change indexs
```

### 3. DataFrame
```
dic = {'cla':[1,2,3,4],'colb':[2,4,6,8],'col3':[3,5,7,9]}
f = DataFrame(dic)
f = (dic,columns = ['id','num','rank'],index = ['a','b','c','d'])
f = DataFrame({'dic1':{1:1,2:4},'dic2':{1:3,2:6}})

f['id']
f.id   # without [ ]
f.ix['a'] # row index with ix
f.ix[1]

f['rank'] = numpy.arange(5)
f['rank'] = Series([1,2,3,4]，index = ['?','/','!'])
del f['rank']

f.columns.name = 'name'
f.index.name = 'info'
f.values

df[val]	
df.ix[val]	
df.ix[:,val]	
df.ix[val1,val2]
```

icol、irow 方法	
get_value、set_value 方法	

### 4.Methods for  Pandas 
describe	针对 Series 或各 DataFrame 列计算汇总统计
mean	值的平均数
count	非 NA 值的数量
min、max	计算最小值和最大值
cumsum	样本值的累计和
argmin、argmax	计算能够获取到最小值和最大值的索引位置（整数）
median	值的算术中位数（50%分位数）
idxmin、idxmax	计算能够获取到最小值和最大值的索引值
sum	值的总和
var	样本值的方差
std	样本值的标准差
diff	计算一阶差分（对时间序列很有用）

quantile	计算样本的分位数（0到1）
mad	根据平均值计算平均绝对离差
skew	样本值的偏度（三阶矩）
kurt	样本值的峰度（四阶矩）
cummin、cummax	样本值的累计最大值和累计最小值
cumprod	样本值的累计积
pct_change	计算百分数变化
axis	简约的轴。DataFrame 的行用0，列用1
skipna	排除缺失值，默认值为 True
level	如果轴是层次化索引的（即 MultiIndex），则根据 level 分组约简

## Scikit-Learn Machine Learning
### 1. concept:
 1.  ML considers how to predict some characteristic of the unknown data by a set of data sample.
 2. classification:
  - supervised learning: 
the sample data carrys the part we want to predict. it can be devided into:
 *Classification*: the sample belongs to two or more classes. we predict the type of unknown data by learning sample of known classes.
*Regression*: hope to output continue variables
- unsupervised learning:
sample data contains no objective value. it aims to mine similar parts from the data to form different groups - cluster(聚类). Or find the density distribution of input space
of data, which means density assumption. Or reduce dimension for data visualization.
 2. use what we learn from **train set** to predict **test set**
### 2. basic operations:
```
from sklearn import datasets
iris = datasets.load_iris()
iris.data.shape  # see how data stored
iris.target.shape  # see how types stored
import numpy as np
np.unique(iris.target)  # show all types
```
```
digit = dataset.load_digits()
print(digit.data)
digit.target
```
###3.learn and predict
in scikit-learn, **estimator** is a python object which implements fix(x,y) and predict(T) method.
class **sklearn.svm.SVC** is an estimator supporting vector classification. we treat it as a black box regardless of the detailed algorithm and chosen of paramaters.
```
from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100.)
# set gammar manually 
# we can find better parameters with node serach(格点搜索) and cross verification(交叉验证)
```
pass the train set to fit method for data fitting. here we take the last one as prediction and the remain as train data
```
clf.fit(digit.data[:-1],digit.target[:-1])
```
the output is 

`SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
`
again
```
clf.predict(digit.data[-1])
```
output:
    `array([8])`

### 4.Regression
in regression model, objective value is the linear combination of input value.
y(w,x) = w0+w1x1+w2x2+...+wpxp
w = (w_1,w_2,...,w_p) is coef_（因数）
w_0 is intercept(截距)
least square method(最小二乘法) minimize residual（残差）
```
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit([[0,0],[1,1,])
clf.coef_
```
output:
`array([0.5,0.5])`

### 5. Classification
the simplest algorithm is nearest-neighbor: given a new data, take the tag of the nearest sample in N-dimension space as its tag.
```
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data,iris.target)
knn.predict([[]0.1,0.2,0.3,0.4])
```
output:
`array([0])`

###6. Cluster
The simpest cluster algorithm is k-means: classify data into k types, allocate a sample into a type according to the distance between the sample and average value of all the current samples of the type. 
With each sample adding in, average value keeps renewing until the type reaches convergence after several turns(determined by `max_iter`).

##Numpy basic operations
`import numpy as np`
###1. numeric types
- bool 
- inti, int8,int16,int32,int64
- uint8,uint16,uint32,uint64
- float,float16,float32,float64
- complex,complex64,complex128
`complex(1,2)`
###2. Numpy array object
- consists of two parts: the actual data and some describing metadata
- basic APIs:
```
a = np.arange(10) # arange(x) used to generate ordered numbers
a = np.array([np.arange(3),np.arange(3)])
a.dtype # check the data type
a.dtype.itemsize 
a.shape # check dimension with tuple
a[2,1] # visit element
```
- one-dimensional slicing and indexing: the same as python list
```
a[3:7]  # slice 
a[:7:2]  # with increment
a[::-1]  # reverse
```
- manipulating array shapes 
reshape() & shape
```
# ar is an numpy array
ma = ar.reshape(2,3,4)
ar.shape = (2,3,4)
# use data to form a multidimensional array of different vectors 
```
resize() 
```
ar.resize(3,81)
# same as reshape() but change the array itself
```
flatten() & ravel()
```
ma.flatten()
ma.ravel()
# push multidimension into one
```
transpose()
`ma.transpose()`

- ###stacking array
-- horizontal stacking:  
`np.hstack((a,b)) # take a tuple as input`
`np.concatenate((a,b),axis=1) # is the same`
-- vertical stacking:
`np.vstack((a,b)) # take a tuple`
`np.comcatenate((a,b),axis=0)`
-- depth stacking: by dimension 
`np.dstack((a,b))`

- ###splitting Numpy array
-- horizontal splitting: into arraies 
```
a = np.array([0,1,2]，[3,4,5]，[6,7,8])
np.hsplit(a,3)
```
-- vertical splitting
```
np.vsplit(a,3)
```
-- depth-wise splitting:each columne from each dimension
```
c = np.arange(27).reshape(3,3,3)
np.dsplit(c,3)
```
### - array attributes: like python property
` # b is an numpy array`
`b.ndim` check the number of dimensions
`b.size` holds the count of elements
`b.itemsize`returns the count of bytes for each element
`b.nbytes` returns the total bytes (=size*itemsize)
`b.T` the same as b.transpose()
complex number can be held like:
`c = array([1.+2.j,3.+4.j])`
`c.real` and `c.imag` show the real and imaginary part respectively
make it flat:
`f = b.flat` gives back a **numpy.flatiter** object which can be used to iterate each element
```
for item in f:
    print item
```
obtain elements with flatiter object:
`b.flat[2]`  `b.flat[[1,3]]`

- ### converting array to list
`b.tolist()`

- ### array views and copies
views in Numpy are not read only and cannot protect the underlying information
```
import scipy.misc
import matplotlib.pyplot as plt
lena = scipy.misc.lena() # a picture
lena.copy()
lena.view()
plt.subplot(221)
plt.imshow(lena)
plt.show()
```
### - fancy indexing 
```
import scipy.misc
import matplotlib.pyplot as plt
lena = scipy.misc.lena() # a picture
xmax = lena.shape[0]
ymax = lena.shape[1]
lena[range(xmax),range(ymax)] = 0  # ??
lena[range(xmax-1,-1,-1),range(ymax)] = 0  # ??
plt.imshow(lena)
plt.show()
```

