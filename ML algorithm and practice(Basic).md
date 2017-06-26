# ML algorithm and practice(Basic)
标签： machine_learning
---

## 1.  矢量编程基础
### 矩阵
- 对象是矩阵的一行，特征是矩阵的一列
例如：

- 词袋列表：若干文本提取出不同的词组成的集合
  词向量空间：文本-词袋构成的整数值矩阵
- 分类/聚类 看作是根据对象特征的相似性和差异性，对矩阵空间的划分
- 预测/回归 看作是根据对象在某种序列（时间）上的相关性，表现特征取值变化的趋势

三个方向的用途：
解线性方程：通过计算距离
方程降次：二次型 升维
变换：维度约简

### 矢量化编程
基于矩阵的基本运算
MATLAB 和python的矩阵运算调用C函数完成  数值运算  并行运算
图形处理器GPU (graphic computing unit)

### numpy 矩阵运算
初始化
```
# initialize a 3*4 matrix
import numpy as np
allZero = np.zeros([3,4])  # 全零矩阵
allOne = np.ones([3,4])   # 全一矩阵
myrandom = np.random.rand(3,4)   # 随机矩阵
myeye = np.eye(3) # 3*3 单位矩阵
# 用二维向量初始化矩阵
myMatrix = mat([[1,2],[2,4],[4,8]])
```
元素运算
```
print matrixA + matrixB
print matrixA - matrixB
print c * matrixA
# the sum of all elements
sum(myMatrix)

# multiply elements at the same positions
# a certain broadcast rule will be applied to expand the matrix if dimension is not match.
multiply(matrixA,matrixB)

# the power of each element
power(matrix,2)
```
矩阵相乘
```
matrixa * matrixB
# multiply of two matrixs use operator *, of matrix elements use multiply()
```
矩阵转置
```
matrix.T
matrix.transpose()
```
其他操作:行列切片/拷贝/比较
```
# show the line and column number
[m,n] = shape(matrix)

# slice by line 
line = matrix[0]

# slice by column
col = matrix.T[0]

# copy method
copied = matrix.copy()

# compare each pair of elements at the same position
# return a bool matrix
matrixA > matrixB
```
### numpy的Linalg库提供线性代数运算
1 行列式
`linalg.det(matrix)`
2 逆矩阵
`linalg.inv(matrix)`
3 秩
`linalg.matrix_rank(matrixA)`
4 解线性方程组
```
# solve A*s = B 
s = linalg.solve(A,B)
```

## 2. 数学基础
现代数学三大根基：概率论（事物可能会在怎样）、数值分析（怎样变化）、线性代数（不同观察维度）

###2.1 相似性的度量 similarity measurement（向量的距离）

- **Euclidean范数**
各元素平方和的开方
$$X = \sqrt{\sum_{i=1}^nx_i^2}$$
`linalg.norm(matrixA)`

- **Minkowski Distance**
$$d=\sqrt[p]{\sum_{i=1}^n |a_i-b_i|^p}$$
when p = 1, d is **Manhattan Distance** (city block distance)
`sum(abs(vectorA-vectorB))`

when p = 2, d is **Euclidean Distance**
`sqrt( (vectorA-vectorB)*((vectorA-vectorB).T) )`

when p -> infinite, d is **Chebyshev Distance**
the same as $$d=max_{i=1}^n(|a_i-b_i|)$$
`abs(vectorA-vectorB).max()`

- **Consine**
describe the difference of direction of two vectors
`cosV12 = dot(vectorA,vectorB)/(linalg.norm(vectorA)*linalg.norm(vectorB))`

- **Hamming Distance** 
between two strings: the minimum times of replacement to transform one into the other
```
tmp = nonzero(vectorA-vecotrB)
tmp2 = shape(tmp) # return [line,column]
tmp2[1]
```

- **Jaccard similarity coefficient**
$$J(A,B)={|A\cap B|\over|A\cup B|}$$
- **Jaccard Distance**
$$J_d = 1 - J(A,B)$$ 
```
from numpy import *
import scipy.spacial.distance as dist
matV = mat(vectorA,vectorB)
dist.pdist(matV,'jaccard')
```

### 2.2概率论

- 样本：矩阵对象
样本空间：全体对象
随机事件：某个对象具有某属性
随机变量：某个属性

- 讨论某个对象属于某个类别的可能性

**贝叶斯公式**
$$P(B|A)P(A) = P(A|B)P(B)$$

- 多元统计： 联合概率分布 与 边缘概率分布

- 特征相关性
**expectation 期望**$$E[X]=\sum_{i=1}^np_ix_i$$
**variance 方差**$$D=E[(X-E[X])^2]=\sum_{i=1}^np_i(x_i-E)^2$$
**covariance 协方差**$$cov(X,Y)=E[(X-E[X])(Y-E[Y])]$$
用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况

**covariance matrix 协方差矩阵**$$cov(X,Y)=E[(X-E[X])(Y-E[Y])^T]$$

**correlation coefficient 相关系数**
$$CC_{xy}={Cov(X,Y) \over \sqrt{D(X)} \sqrt{D(Y)}}$$
取值范围[-1,1] -1线性负相关  1线性正相关

**corelation distance 相关距离**
$$D_{XY}=1-CC_{XY}$$

```
# 均值
m1 = mean(vectorA)
# 标准差 = 方差的平方根
dv1 = std(vectorA)
# 相关系数
corref = mean(multiply(vectorA-m1,vectorB-m2))/(dv1*dv2)
# 相关系数矩阵
print corrcoef(mat(vectorA,vectorB))
```

**Mahalanobis Distance马氏距离**
排除量纲对相关性的干扰
$$有M个样本向量X_1,X_2,...X_m,协方差矩阵为S,均值为E$$
$$M(X) = \sqrt{(X-E)^TS^{-1}(X-E)}$$
$$M(X_i,X_j) = \sqrt{(X_i-X_j)^TS^{-1}(X_i-X_j)}$$
```
matrix = mat(vectorA,vectorB)
covinv = linalg.inv(cov(matrix))
tp = vectorA-vectorB
mditance = sqrt(dot(dot(tp,covinv),tp.T)) 
```

### 2.3 线性空间变换
向量乘矩阵：向量从一个线性空间变换到另一个线性空间的过程
矩阵乘矩阵：向量组的空间变换，维度对齐
一组特征向量：变换过程只发生伸缩、不发生旋转
特征值：伸缩的比例
```
# 特征值 特征向量
eval,evec = linalg.eig(matrix)
# 还原
matrix = evec*(eval*eye(m))*linalg.inv(evec)
```

### 2.4 数据归一化
变成（0,1）之间的小数，或者有量纲变成无量纲

- 标准化欧式距离：各个分量都标准化到均值方差相等
- 以方差的倒数为权重，**加权欧氏距离**
$$d = \sqrt{\sum_{k=1}^n({X_{1k}-X_{2k} \over S_k})^2}$$

### 2.5数据处理与可视化
- 读取
从文件读取数据形成矩阵
```
def fileToMatrix(path,delimiter): # 'path' is the path of a file
    list = []
    fp = open(path,"rb")
    content = fp.read()
    fp.close()
    rowlist = content.splitlines()
    recordlist = [row.split(decimiter) for r in rowlist if r.strip()]
    return mat(recordlist)
```
- 可视化
```
import matplotlib as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X,Y,c='red',marker='o') # draw dots
ax.plot(X,y,'r') # draw curve
plt.show()
```
http://matplotlib.org/examples/index.html

## 3. 中文文本分类
总体步骤：
预处理--中文分词--构建词向量空间--权重策略--使用分类器--评价结果
### 3.1 预处理
- 选范围
- 建立语料库： 训练集  测试集
- 文本格式转换： 转为TXT或者XML
过滤标签  小心原本就有的乱码
python去除HTML标签使用lxml库，使用海量网络文本转换
- 检测句子边界：
英文句号容易跟缩写混淆，可以使用简单的启发式规则和统计分类技术
（图片式标点必须被字符型替代）

### 3.2 中文分词 Chinese Word Segmentation
词没有形式上的分界符    
NLP的核心问题?

中文分词算法： 基于概率图模型的条件随机场(CRF)——Lafferty

文本结构化模型： 词向量空间模型、主题模型、依存句法的树表示、RDF的图表示

jieba分词系统 使用CRF算法和python
```
# 返回可迭代的generator
# 默认切分
seg_list = jieba.cut("中文文本串",cut_all=False)
# 全切分
seg_list = jieba.cut("中文文本串",cut_all=False)
# 搜索引擎粒度切分
seg_list = jieba.cut_for_search("中文文本串")

print "/".join(seg_list)
list(seg_list)
```
more use: https://www.oschina.net/p/jieba


- 语料库分词
无法排除全角空格出现在文件中导致无法read()的问题
因为str没有了decode()方法
这里采用无视全角空格的文件的方法
```
import sys 
import os
import jieba
import codecs
 #reload(sys) 被淘汰的用法
 #sys.setdefaultencoding('unicode')

def savefile(path,content):
    fp = open(path,"w")
    fp.write(content)
    fp.close()

def readfile(path):
    fp = open(path,"r")
    try:
        content = fp.read()
    except:
        content = ""
    fp.close()
    return content

corpus_path = "D:/Data Science Experiment/Chinese Text Clustering/train_corpus/"
seg_path = "D:/Data Science Experiment/Chinese Text Clustering/corpus_segments/"

# get all the files under this path
catelist = os.listdir(corpus_path)
print(catelist)
for mydir in catelist:
    
    class_path = corpus_path + mydir + "/"
    seg_dir = seg_path + mydir + "/"
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)
    print(file_list)
    for file_path in file_list:
        print(file_path)
        # make up the path of target txt
        fullname = class_path+file_path
        content = readfile(fullname).strip()
        content = content.replace("\r\n","").strip()
        content_seg = jieba.cut(content)
        savefile(seg_dir + file_path, "/".join(content_seg))
```

- 转化为Bunch类储存
os, pickle, bunch 的使用
```
import sys
import os
import pickle
from sklearn.datasets.base import Bunch

bunch = Bunch(target_name = [],label=[],filename=[],contents=[])

seg_path = "D:/Data Science Experiment/Chinese Text Clustering/corpus_segments/"
bunch_path = "D:/Data Science Experiment/Chinese Text Clustering/words_bag.dat"

catelist = os.listdir(seg_path)
print(catelist)
for mydir in catelist:
  class_path = seg_path + mydir + "/"
  file_list = os.listdir(class_path)
  for file_path in file_list:
    fullname = class_path + file_path
    bunch.label.append(mydir)
    bunch.filename.append(fullname)
    fileObj = open(fullname)
    bunch.contents.append(fileObj.read().strip())
    fileObj.close()

file_obj = open(bunch_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()
print("finished!")
```

- Scikit-learn
http://scikit-learn.org/stable/
分类与回归算法、聚类算法、维度约简、模型选择、数据预处理
有教程，多看

- 向量空间模型
把文本储存为向量 缺点是维度会很高
过滤一些**停用词**
下载停用词表

### 3.3 权重策略： TF-IDF方法
**词频Term Frequency**
是指某个给定词语在某文件中的出现频率。（考虑重复计数）
$$TF_w = {w在文件的出现次数 \over 文件词语出现总数}$$
**逆向文件频率IDF**
代表一个词语的普遍重要程度
$$IDF_w = log{文件总数 \over 1 + 包含词语某个w的文件数}$$
**TF_IDF**

$$TF_w*IDF_w$$
可以衡量某个词语w的重要性
某一“重要”的词语在某一文件内高频、在全部文件中低频，则可以用来分类，TF-TDF权重就高。

```
from sklearn.datasets.base import Bunch
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def readBunchObj(path):
  fileObj = open(path,"rb")
  bunch = pickle.load(fileObj)
  fileObj.close()
  return bunch

def writeBunchObj(path,bunchObj):
  fileObj = open(path,"wb")
  pickle.dump(bunchObj,fileObj)
  fileObj.close()

path = "D:/Data Science Experiment/Chinese Text Clustering/words_bag.dat"
bunch = readBunchObj(path)

tfispace = Bunch(target_name = bunch.target_name,label=bunch.label,filename=bunch.filename,tdm=[],vocabulary=[])

vectorizer=TfidfVectorizer(max_df=0.5)
tansformer = TfidfTransformer()

tfispace.tdm=vectorizer.fit_transform(bunch.contents)
tfispace.vocabulary = vectorizer.vocabulary_

space_path = "D:/Data Science Experiment/Chinese Text Clustering/tfidf_space.dat"
writeBunchObj(space_path,tfispace)

print("finished!")
```

### 3.4 文本分类方法
1. kNN最邻近方法：简单、精度一般、速度慢
2. 朴素叶贝斯算法：对短文本分类效果好、精度高
3. 支持向量机算法：支持线性不可分的情况？

朴素叶贝斯算法实现
```
class bayes(obj):
  def _init_(self):
    self.vocab = []
    self.idf = 0 # matrix
    self.tf = 0 # matrix
    self.tdm = 0 # P(x|y)
    self.Pcates = {}
    self.labels = []
    self.doclenth = 0
    self.vocablen = 0
    self.testset = 0
    
  def train_set(self,trainset,classvec):
    self.cate_pro(classvec)
    self.doclenth = len(trainset)
    
    tempset = set()
    [tempset.add(word) for doc in trainset for word in doc]
    self.vocab = list(tempset)
    
    self.vocablen = len(self.vocab)
    self.cul_word_freq(trainset)
    self.build_tdm()
    
  def cate_pro(self,classvec):
    self.labels = classvec
    labeltmp = set(self.labels)
    for label in labeltmp:
      self.Pcates[label] = float(self.labels.count(label)/float(len(self.labels)))
    
  def cul_word_freq(self,trainset):
    self.idf = np.zeros([1,self.vocab])  # build two matrics? 
    self.tf = np.zeros([self.doclenth,self.vocablen])
    for f in range(self.doclenth):
      for word in trainset[f]:
        self.tf[f,self.vocab.index(word)]+=1
      for signalword in set(trainset[f]):
        self.idf[0,self.vocab.index(signalword)]+=1
  
  def build_tdm(self):
    self.tdm = np.zeros([len(self.Pcates),self.vocablen])
    sumlist = np.zeros(len(self.Pcates),1)
    for index in range(self.doclenth):
      self.tdm[self.labels[index]] = np.sum(self.tdm[self.labels[index])
    self.tdm = self.tdm/sumlist
    
  def mapTolist(self,testdata):
    self.testset = np.zeros([1,self.vocablen])
    for w in testdata:
      self.testset[0,self.vocab.index(w)]+=1
  
  def predict(self,testset):
    if np.shape(testset)[1] != self.vocablen:
      print("dimension error")
      exit(0)
    else:
      prevalue = 0
      preclass ""
      for tdm_vec, keyclass in zip(self.tdm,self.Pcates):
        tmp = np.sum(testset * tdm_vec * self.Pcates[keyclass])
        if tmp > prevalue:
          prevalue = tmp
          preclass = keyclass
    return preclass
```

### 3.5 分类结果评估
机器学习算法评估三指标：
**召回率Recall Rate**
相关文件里被检索到的比例
$$Recall = {检索到的相关文件数 \over 系统所有相关文件数}$$
**准确率（精度）Precision**
检索到的文件里相关文件的比例
$$Precision = {检索到的相关文件数 \over 所有检索到的文件数}$$
**F-Score**
$$F_t={(t^2+1)PR \over t^2P+R}$$
t是参数，P是Precision,R是Recall
当t=1时是最常见的F1-Measure
P和R的调和平均数
$$F_1 = {2PR \over P+R}$$

### 3.6 朴素贝叶斯分类算法
公式推导略

### 3.7 KNN分类算法
k-nearest neighbor
A simple ML algorithm that do classification by measuring distances of different features
If most of the K-nearest neighbors of a sample in the feature space belong to  a certain class, the sample belongs to it.
Steps:
1. determine the value of K.(often an odd number)
2. determine the formula to calculate distance.(for text classification consine is often used) select the nearest K samples.
3. calculate the number of different classes in the K samples. The max is what we look for.

```
def cosine(vec1,vec2):
  v1 = np.array(vec1)
  v2 = np.array(vec2)
  return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# return the predited label of test_vec
def classify(test_vec, train_set, list_class, k):
  data_set_size = len(train_set)  
  distance = np.array(np.zeros(data_set_size)) # 1-D
  for index in range(data_set_size):
    distance[index] = cosine(test_vec,train_set[index])
  sorted_dist = np.argsort(-distance)
  print(sorted_dist)
  class_cnt = {}
  for i in range(k):
    label = list_class[sorted_dist[i]]
    class_cnt[label] = class_cnt.get(label,0) + 1
  sorted_class_cnt = sorted(class_cnt.items(), key=lambda d:d[1], reverse = True)
  print(sorted_class_cnt)
  return sorted_class_cnt[0][0]
```


## 4 决策树
（CLS学习系统--ID3算法--ID4算法--C4.5算法&CART算法）
### 4.1 决策树的基本思想
**CLS(Concept Learning System)**: the fundation of decision trees
there are three kinds of tree nodes: root, leaf and inner node.(根，叶子，内点)
Steps:
1. Start from an empty tree. randomly select the first feature as the root.
2. Do classification according to a certain condition. If the sub-set classified is empty or labeled the same, this sub-set is leaf. Otherwise, it is an inner node.
3. If it is an inner node, select a new label/feature to classify until all sub-sets are leaves.

根节点和内点是特征，叶子是结果判断
树枝是该特征取某个值的概率

### 4.2 决策树的算法框架
#### 1. 主函数
递归函数，负责节点生长和结束算法

- 输入需要分类的数据集和已知标签
- 调用“计算最优特征子函数”：根据某种规则确定最优划分特征，创建特征节点
- 调用“划分函数”：按照该特征把数据集划分为若干部分
- 构建新的分支节点
- 检验递归终止条件
- 将新节点的数据集和标签作为输入，递归执行以上步骤

#### 2. 计算最优特征子函数
不同决策树的根本差异
遍历当前数据集，评估每个特征，返回最优者

#### 3. 划分函数
分割数据集，删除特征

#### 4. 分类器
分类或预测

### 4.3 信息熵测度
- 特征离散化：把字符串用整数表示
- 选取**无序度**最大的特征作为划分节点
- 信息： 对不确定性的消除，从信源的消息转换而成的状态，是随机事件。
信源发送什么信息是不确定的，概率大，出现机会多，不确定性就小。
- 熵Entropy：任何一种能量在空间中分布的均匀程度

- 不确定性函数I称为事件U的信息量，是概率的递减函数
$$I(U)=-log(p)$$
- 信源事件有n种取值，对应概率为pi，各事件彼此独立。信源的平均不确定性，也称为**信息熵**
$$H(U)=E[-log(p_i)]=-\sum_{i=1}^np_ilog(p_i)$$
取2为底，就是信息单位bit
某个特征列向量的信息熵大，说明混乱程度高，应优先考虑划分。
**信息熵**为决策树的划分提供最重要的依据

- 设n个数据的集合为S，具有m个标签。每个标签定义不同的类Ci(i=1-m)，设ni为类Ci中的样本数,pi=ni/n。S的信息熵为
$$I(s_1,s_2,..,s_m)=-\sum_{i=1}^mp_ilog_2(p_i)=-\sum_{i=1}^m\frac {n_i}n log_2(\frac{n_i}n)$$

- 设特征A具有v个不同值，用特征A将S划分为v个子集{s1,s2...sv}，即v个决策树分支，设nij为子集sj中属于类Ci的样本数。集合S根据A划分之后的信息熵为
$$E(A)=\sum_{j=1}^v\frac{\sum_{i=1}^m s_{ij}}nI(s_{ij},s_{2j}...s_{mj})$$
$$=\sum_{j=1}^v\frac{\sum_{i=1}^m s_{ij}}n (-\sum_{i=1}^mp_{ij}log_2(p_{ij}))$$
$$=\sum_{j=1}^v\frac{\sum_{i=1}^m s_{ij}}n (-\sum_{i=1}^m \frac {s_{ij}}{|s_j|}log_2(\frac{s_{ij}}{|s_j|}))$$

- 信息增益：由于知道属性A的值/根据A做出划分，导致的信息熵的期望压缩
$$Gain(A)=I(s_1,s_2,..,s_m)-E(A)$$

### 4.3 ID3 Decision Tree
#### ID3 algorithm
- culculate information entropy of the given unsorted sample
- culculate information entropy of each feature
- select the feature with the largest information gain as a root or inner node
- divide the dataset into different sub-sets according to different value of the feature.delete the current feature column. culculate information entropy of the rest. if there is information gain, repeat the steps above.
- if there is only one feature label in a sub-set, stop division.

#### code
```python
class ID3_DTree(object):
  def _init_(self):
    self.tree = {}
    self.dataset = []
    self.labels = []
    
  def train(self):
    labels = copy.deepcopy(self.labels)
    self.tree = self.buildTree(self.dataset,labels)
    
  """ 
  dataSet input instruction:
    row for object, column for feature
    the last column is class
  """  
  def buildTree(self,dataset,labels):
    cateList = [data[-1] for data in dataset]  # pick up all the classes
    if cateList.count(cateList[0]) == len(cateList): # only one feature
      return cateList[0]
    if len(dataset[0]) == 1: # no feature!
      return self.maxCate(cateList)
    bestFeat = self.getBestFeat(dataset)
    bestFeatLabel = labels[bestFeat]
    tree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # take this column
    uniqueVals = set([data[bestFeat] for data in dataset])
    for value in uniqueVals:
      subLabels = labels[:] # deep copy
      splitDataSet = self.splitDataSet(dataset,bestFeat,value)
      subTree = self.builTree(splitDataSet,subLabels)
      tree[bestFeatLabel][value] = subTree
    return tree
    
    """
      culculate the label that appears mostly
    """
    def maxCate(self,cateList):
      items = dict([(cateList.count(i),i) for i in cateList])
      return items[max(items.keys())]


    def getBestFeat(self,dataset):
      numFeat = len(dataset[0])-1
      baseEntropy = self.computeEntropy(dataset)
      bestInfoGain = 0.0
      bestFeat = -1
      for i in range(numFeat):
        uniqueVals = set([data[i] for data in dataset])
        newEntropy = 0.0
        for value in uniqueVals:
          subDataSet = self.splitDataSet(dataset,i,value)
          prob = len(subDataSet)/float(len(dataset))
          newEntropy += prob * self.computeEntropy(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
          # painter's method
          bestInfoGain = infoGain
          bestFeat = i
        return bestFeat
    
    def computeEntropy(self,dataset):
      datalen = float(len(dataset))
      cateList = [data[-1] for data in dataset]
      items = dict([(i,cateList.count(i)) for i in cateList])
      infoEntropy = 0.0
      for key in items:
        prob = float(items[key])/datalen
        infoEntropy -= prob * math.log(prob,2)
      return infoEntropy
      
    # abundon all the unit in column['axis'] with 'value' 
    # so that te dataset could be compressed.
    def splitDataSet(self,dataset,axis,value):
      rList = []
      for featVec in dataset:
        if featVec[axis] == value:
          rFeatVec = featVec[:axis]
          rFeatVec.extend(featVec[axis+1:])
          rList.append(rFeatVec)
      return rList
      
     def predict(self,testVec):
      root = self.tree.key()[0]
      secondDict = self.tree[root]
      featIndex = self.labels.index(root)
      key = testVec[featIndex]
      valueOfFeat = secondDict[key]
      if isinstance(valueOfFeat,dict): # judge whether it is a dictionary
        classLabel = self.predict(valueOfFeat,self.labels,testVec)
      else:
        classLabel = valueOfFeat
      return classLabel
```
- flaws
信息增益偏向选择特征值个数较多的特征，取值个数多不一定最优。
大型数据会生成层次和分支很多的决策树，其中某些分支的特征值概率很低，如果不加忽略就会过分拟合。

### 4.4 C4.5算法
用 信息增益率 代替信息增益
$$GainRatio(S,A)=\frac{Gain(S,A)}{-\sum_{i=1}^mp_{ij}log_2(p_{ij})}$$
$$ = {-\sum_{i=1}^m\frac {n_i}n log_2(\frac{n_i}n)-\sum_{j=1}^v\frac{\sum_{i=1}^m s_{ij}}n (-\sum_{i=1}^m \frac {s_{ij}}{|s_j|}log_2(\frac{s_{ij}}{|s_j|})) \over -\sum_{i=1}^m \frac {s_{ij}}{|s_j|}log_2(\frac{s_{ij}}{|s_j|}) }$$

### 4.5 回归树
#### 回归算法原理
Classification and regression tree(CART)成熟广泛应用
通过决策树实现回归
回归模型中，样本取值分为观察值和输出值，都是**连续**的。
CART使用最小剩余方差(Squared Residuals Minimization)判定回归树的最优划分.使用线性回归模型进行建模。如果难以拟合，继续划分子树。直到所有叶子节点都是线性回归模型。

#### 最小剩余方差(Squared Residuals Minimization)
二重循环遍历每个特征列的所有样本点，在每个样本点上二分数据集，计算出最小的总方差（划分后的两个子集总方差之和）
总方差是每个数据与均值的方差的和。

#### 模型树
叶子是一系列分段线性函数，是对原数据曲线得到模拟。
模型树还有很多性质

#### 剪枝
连续数据会生出大量分支，需要对预测树剪枝
先剪枝：预定义划分阀值，低于阀值划分停止
后剪枝：计算内点的误判率，当子树的误判个数减去标准差后大于对应叶子节点的误判个数，就决定剪枝。

#### Scikit-Learn 对回归树的实现

## 5 推荐系统原理
### 5.1
经常一起购买的商品（打包销售）、购买此商品的客户也同时购买（协同过滤）、看过此商品后的客户买的其他商品、商品评分列表、商品评论列表

推荐系统的架构 （图）

推荐算法
- 基于人口统计学的推荐机制
- 基于内容的推荐
- 基于协同过滤的推荐：基于用户、基于项目
- 基于隐语义/模型的推荐：SVD隐语义模型

### 5.2 协同过滤
CF Collaborative Filtering
分为基于用户和基于项目：找到具有类似品味的人喜欢的物品、找到与喜欢的物品类似的物品

#### 1. 数据预处理
对用户行为分组、加权
减噪（利用减噪算法）、归一化（统一量纲，除以最大值）
进行聚类，降低计算量

#### 2. 使用scikit-Learn的KMeans聚类
给定划分的个数k。
创建初始划分，随机选择k个对象，各自代表聚类中心。其他对象属于离它最近的聚类。
迭代。当有新的对象加入或者离开某聚类时，重新计算聚类中心，然后重新分配。不断重复，直到没有聚类中的对象变化。
```
from sklearn.cluster import KMeans
kmeans = KMeans(init='k-mean++',n_cluster=4)
kmeans.fit(dataMatrix)
kmeans.cluster_centers_
```

#### 3. User CF 原理
User Item矩阵：行是用户列表，列是物品列表，矩阵值是用户偏好数值。

用户相似度矩阵：按行归一化，一行总和为1

- 一个用户的偏好是一个向量，利用聚类算法，基于用户对物品的偏好划分用户类型；
- 用KNN邻近算法找到最邻近的用户，根据相似度权重和对物品的偏好，预测当前用户可能有偏好的物品
- 用户甲与用户乙相似，则将用户乙购买的商品推荐给用户甲

相似度的评判使用距离函数：
欧氏距离、相关系数、Jaccard距离、余弦距离

#### 4. Item CF 原理
应用普遍广泛
物品相似度矩阵：按列归一化，一列的和为1

- 根据用户偏好划分物品类型（聚类算法），计算物品之间的相似度（KNN算法），找最邻近的物品，预测当前用户可能有偏好的物品
- 物品A与物品B相似，则将物品A推荐给购买物品B的用户

问题：人为分类对推荐算法造成影响；物品相似度受个人消费影响，用户难以加权
需要一种算法针对每类用户的不同消费行为计算不同物品的相似度

#### 5. SVD 原理
隐语义模型（奇异值分解，SVD）通过隐含特征计算相似性
Singular Value Decomposition
The singular value docomposition of a m*n real or complex matrix $M$ is the factorization of the form $U\sum{}V^T$, where $U$ is a m*m orthonormal matrix, $\sum{}$ is a rectangle diagonal matrix with no negative value in diagonal, and $V$ is a n*n orthonormal matrix. 
The diagnoal entries of $\sum{}$ is the **singular values** of $M$.
$U$ is called the left-singular vector. $V$ is called the right-singular vector.
$U$ is a set of orthonormal eigenvectors of $MM^T$.
$V$ is a set of orthonormal eigenvectors of $M^TM$.  So it can be got by solving $(MM^T)V=\lambda{}V$.
$s = \sqrt{\lambda}$
$U = \frac{MV}{s}$

```
import numpy as np
U,s,V = np.linalg.svd(matrix)
# s is in decending sort, so you cannot find out the unsorted s 
```
but why it works?
奇异值在$\sum{}$中按降序排列，衰减特别快，前10%甚至1%的奇异值占了奇异值总和的99%，所以可以用前几个奇异值来近似描述矩阵。奇异值由矩阵本身唯一决定.

Partial SVD（部分奇异值分解）：
$M_{m\times n} \approx U_{m\times r}\sum_{r\times r}V_{r\times n}$
where r<< m and n
储存UsV可以节省空间

DIY SVD
```
import numpy as np
def SVD(M):
    lam,hU = np.linalg.eig(Matrix*Matrix * M.T)
    eV, hVT = np.linalg.eig(Matrix.T*Matrix.T * M)
    hV = hVT.T
    sigma = np.sqrt(lam)
    return hU, sigma, hV
```

分解之后，U,s,V都取前r个值，计算出待测算的向量。
```
U,s,V = np.linalg.svd(dataset.T)
V = V.T
Ur = U[:,:r]
Sr = np.diag(s)[:r,:r]
Vr = V[:,:r]
testResult = testVec * Ur * np.linalg.inv(Sr)
result = array([dist(testResult,vi) for vi in Vr])
descIndex = argsort(-result)[:r]
```

### 5.3 KMeans评估
簇：有距离相近的对象组成

- 从N个点中随机选取K个作为质心
- 测量剩余文档到质心的距离，归入最近的类
- 重新计算各类质心
- 迭代上述步骤。直到新质心与原质心相等或者距离不超过阀值。

KMeans不总是能找到正确的聚类。
KMeans擅长处理球状分布的数据，当类与类之间的区别比较明显时，效果较好。
复杂度为O(nkt)，n是对象个数，k是簇的个数，t是迭代次数。

问题：
初始点选择影响迭代次数或者限于某个局部最优状态（？）
K要事先给出，不同数据集之间没有可借鉴性
对噪声和孤立点敏感

### 5.4 二分KMeans算法
Bitseting KMeans

- 将所有点作为一个簇，一分为二
- 选择能够最大限度降低聚类代价函数（误差平方和）的簇，一分为二
- 以此进行下去，直到簇的数目等于给定数目K

原理：聚类的误差平方和越小，数据点越接近质心，越密集；误差平方和越大，有可能多个簇被划分为一个。

代码略




