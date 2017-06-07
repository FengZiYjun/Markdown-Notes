# ML algorithm and practice(Advanced)
## 六、梯度寻优
### 6.1 最优化与计算复杂性
最优化理论试图解决多元化、非线性的问题。最早的优化方法是线性规划。
最优化理论三个重要基础： 

- 矩阵理论
- 数值分析
- 计算机

解决最优化问题需要三种能力：

- 数学建模
- 公式推导
- 算法设计

最优化：在给定约束条件下寻求某些量，使得某些量达到最优。
$$min_{x\in{R}}f(x),(maxf(x)) s.t.h_i(x)=0, g_j(x)<=0$$
x为决策变量，$f(x)$为目标函数或cost function。
最优化问题分为：
线性规划（都是线性函数）
非线性规划（至少有一个是非线性函数）
二次规划（目标函数为二次，约束函数为线性）
多目标规划（目标函数是向量函数）等等

### 6.2 凸集与分离定理
1. 凸集定义
 设$X\subseteq{R^n}$，$X$是凸集，当且仅当：
$\alpha x_1+(1-\alpha )x_2 \in{X}, \forall x_1,x_2\in{X},\forall \alpha\in{[0,1]}$
如果一个集合是凸集，那么任意两个点连线上任意一点也位于该集合中。

2. 超平面定义
$X = \{x|c^Tx=z\}$
c为系数向量，z是标量，集合X为超平面。
超平面能把凸集分为两部分$c^Tx\leq z$和$c^Tx\geq z$  
**支撑超平面**：过凸集的一个边界点，使得凸集所有点都位于超平面的一侧

3. 凸集分离定理
设$S_1,S_2\subseteq{R^n}$,如果存在非零向量$p\in{R^n}$以及$\alpha\in{R}$，使得
$$S_1\subseteq{H^-}=\{x\in{R^n}|p^Tx\leq\alpha\}\\
S_2\subseteq{H^+}=\{x\in{R^n}|p^Tx\geq\alpha\}$$
则称超平面$H=\{x\in{R^n}|p^Tx=\alpha\}$分离了集合$S_1$和$S_2$.

分离定理为机器学习的分类问题提供理论依据。带有不同标签的训练集是凸集，分隔它们的平面就是线性分类器。

###6.3 凸函数
若定义某个向量空间的凸子集上的函数$f$，对于任意两点$x_1,x_2$以及$\alpha \in{[0,1]}$，都有
$$f(\alpha x_1+(1-\alpha)x_2)\leq f(\alpha x_1)+f((1-\alpha)x_2)$$
则函数$f$是凸函数。
任意两点的线性组合的函数值小于任意两点函数值的线性组合。
因为凸集元素可以是离散的，所以凸函数不要求连续。

凸函数的判定：
1. 若$f(x)$是凸函数，则$\alpha f(x)$也是凸函数$(\alpha\geq0)$。
2. 若$f_1,f_2\dots f_k$是凸集$S$上的凸函数，则$\sum_{i=1}^k\lambda_if_i(x),\forall\lambda_i\geq0$和$\max_{1\leq i\leq k}f_i(x)$也是$S$上的凸函数。
3. 设$f(x)$在凸集$S$上可微，则$f(x)$为凸函数的充要条件是，对任意的$x_,y$都有$$f(y)\geq f(x)+\nabla f(x)^T(y-x) $$其中$\nabla f=grad f$
4. 设在开凸集$D\subseteq{R^n}$内，$f(x)$二阶可微，则$f(x)$为$D$的凸函数的充要条件是，对任意的$x\in{D}$，$f(x)$的hesse矩阵半正定。$$G(x)=\nabla ^2f(x)=二阶偏导矩阵$$

常用凸函数：  

- 线性函数和仿射函数（一阶多项式函数）
- 最大值函数
- 幂函数、绝对值幂函数
- 对数函数、指数和的对数函数
- 几何平均
- 范数

凸函数的性质：

- 凸优化的任一局部极小值也是全局极小值，且全局极小点的集合为凸集？
- 凸优化的任一局部最优解都是它的全局最优解？

机器学习使用最优化方法的目标函数几乎都是凸函数。无法转化为凸函数的优化问题用穷举法或者一些随机优化方法解决。

### 6.4 计算复杂性
自动机常指基于状态变化进行迭代的算法。

**确定性**：根据输入和当前状态，自动机的状态转移是唯一确定的。
程序下一步的结果是唯一的，返回的结果是唯一的。

**非确定性**：在每一时刻有多种状态可供选择，并尝试执行每个可选择状态。
运行时每个执行路径是并行的，所有路径都可能返回结果；只要有一个路径返回结果，算法就结束。
在求解最优化问题时，非确定性算法可能陷入局部最优，不一定是全局最优。

**P类**问题：能够以多项式时间的确定性算法对问题进行判定或求解。
**NP类**问题：用多项式时间的非确定性算法来判定或求解
**NP-complete**问题：至今没有找到多项式时间的算法

典型的NP类问题：

- 背包问题。一种组合优化的NP完全问题。
- 最短路径问题
- 货郎担问题(Travelling Sales Person问题)。找最短哈密顿回路。属于多局部最优问题。用模拟退火算法。
- 最大团问题。给定无向图和正整数K，是否存在具有K个顶点的完全子图。在概率图模型中，求取最大团是典型的智能推理算法。
- 图同构问题。在自然语言中，图同构是知识推理的一部分。

### 6.5 迭代法解方程组
针对大型稀疏矩阵方程组

将原方程$f(x)=0$化为等价形式$$x=F(x)$$取一个初值$x_0$，按照$x_{k+1}=F(x_k),k=0,1,2\dots $产生序列$\{x_k\}$。这样的计算过程成为迭代。
迭代过程中，根据相邻两次迭代值是否满足精度要求，决定迭代是否继续。

从泰勒公式推导出牛顿迭代法(牛顿切线法)：$$F(x)=x-\frac{f(x)}{f'(x)}$$切线法收敛速度为平方。
如果求导不容易，可以考虑割线法，收敛速度为黄金1.618。用${f(x_k)-f(x_{k-1})\over x_k-x_{k-1}}$代替$f'(x)$。

定理：设$f(x)$在[a,b]中有二阶连续导数，且满足条件
(1) $f(a)*f(b)< 0 $
(2) $f'(x)$ 在(a,b)保号
(3) $f''(x)$ 在(a,b)保号
取(a,b)中满足$f(x_0)*f''(x_0)>0$的一点$x_0$, 以它为初值的牛顿迭代过程产生的序列$\{x_k\}$单调收敛于方程$f(x)=0$在[a,b]中的唯一解。

对于矩阵方程$Ax=b$，使用迭代法 $x^{(k+1)}=B_0x^{(k)}+f,  k=0,1,2\dots$
迭代法收敛的条件是，$\lim_{k\to\infty}x^{（k）}$存在

```
def iteration_solve(B0,f):
    # assert(column of B0==row of f)
    error = 1.0e-6
    steps = 100
    r,n = shape(B0)
    xk = np.zeros([n,1])
    errorlist = []
    for k in range(steps):
        xk1 = xk
        xk = B0 * xk + f
        errorlist.append(np.linalg.norm(xk-xk1))
        if errorlist[-1] < error:
            print(k+1)
            break
    return xk, errorlist
```

问题: 如果目标函数是非线性的，误差往哪个方向下降最快？
函数导数的方向，多元微积分中的梯度。
### 6.6 Logistic梯度下降法

#### 1. 梯度下降法
求解无约束多元函数值极值的数值方法
沿梯度方向向量值函数变化速率最快
为了求取$f(x)$的极小值，任取一个点$x_0$，设$\rho_k$为第k次迭代时的步长
$$\nabla^{(k)}=-{\nabla f(x_k) \over ||\nabla f(x_k)||}\\
x_{k+1}=x_k + \rho_k \nabla^{(k)}$$
产生序列$\{x_k\}$，收敛于$f(x)$极小值。
$\rho$的选择影响算法收敛速度。过小会很慢，过大会发散。

#### 2. 线性分类器
把代表训练集的两个互不相交的凸集的子集分开的支撑超平面，如果是一个n维的线性方程，就称为**线性分类器**。
也是最早得到神经网络模型，叫感知器模型。
感知器 = 算法框架 + 激活函数

- 算法框架$f(x) = w^Tx + b$
- 激活函数$Logistic$函数

Logistic函数：
$$logistic(x) = {1 \over 1+e^{-w^Tx}}$$
是凸函数，局部最优就是全局最优。

若输出标签Y为{0,1}，令$P\{Y=1|x\} = p(x) = {1 \over 1+e^{-w^Tx}}$,那么$P\{Y=0|x\} = 1-p(x) = {1 \over 1+e^{w^Tx}}$,$$ln{\frac{P\{Y=1|x\}}{P\{Y=0|x\}}}=ln{p(x)\over 1-p(x)}=w^Tx$$
为什么权重乘样本矩阵是梯度？？？

假设样本集之间互相独立，它们的联合分布可以表示为各边际分布的乘积。用似然函数表示$$l(w)=\prod_{i=1}^n(P\{Y=1|x_i\})^{y_i}(1-P\{Y=1|x_i\})^{1-y_i}$$
取对数似然函数$$L(w)=ln(l(w))=\sum_{i=1}^ny_iw^Tx_i - \sum_{i=1}^nln(1+e^{w^Tx_i})$$
这是线性分类器的目标函数，以权重向量w为自变量

对w求偏导$$\frac{\partial L(w)}{\partial w} = \sum_{i=1}^n{(y_i-{1 \over 1+e^{-w^Tx_i}})}x_i$$
就是误差函数error = classLabel - logistic

#### 3. 算法流程
输入：自带分类标签的样本矩阵$x$
预处理：提取classLabel，归一化，插入第一列（？）
调参数：迭代次数steps，梯度下降的步长$\alpha$
初始化：权重向量$w$全0
训练分类器：当前结果与分类标签比对，生成误差向量: error = classLabel - logistic,通过迭代修正误差$$w_{k+1}=w_k＋\alpha \times x^T \times error $$
输出：权重向量$w$，作为分类器$f(x) = w^Tx + b$的参数
```
weights = np.ones([n,1])
for k in range(steps):
    gradient = dataMatrix * mat(weights)
    output = logistic(gradient)
    errors = classLabels - output
    weights = weights + alpha * dataMatrix.T * errors
```
权重向量$w$代表分隔空间的超平面。超平面的方程由权重向量$w$决定。

其实这只是一种回归？？？

算法分析：
考察超平面的各参数或者权重向量的各分量是否达到“平稳”，若否，需要增加迭代次数。（或优化模型）

问题：步长取值如何平衡收敛速度和精度的矛盾？

### 6.7 随机梯度下降法
引入随即样本抽取方式，提供动态步长取值，平衡精度和收敛速度
$\alpha = {2 \over 1.0+i+j}+0.0001$
i是迭代次数，j是抽取次数，2与样本均值相关。
改矢量编程为标量编程，用效率的降低换取迭代次数的显著下降
```
for j in range(steps):
    index = range(rowsNum)
    for i in range(rowNum):
        # step length is changed with i and j
        alpha = 2/(1.0+i+j)+0.0001
        # select an index randomly
        randIndex = int(random.uniform(0,len(index))) 
        vecSum = sum(dataMatrix[randIndex]*weights.T)
        grad = logistic(vecSum)
        errors = classLabel[randIndex] - grad
        weights = weights + alpha * errors * dataMatrix[randIndex]
        del(index[randIndex])
```
怎样选择alpha？靠经验

算法评估：
超平面的参数、权重向量各分量可能经历震荡，然后平稳。迭代次数相比原来较小。