# 深度学习

深度学习是机器学习的一个分支。 许多传统机器学习算法学习能力有限，数据量的增加并不能持续增加学到的知识总量，而深度学习系统可以通过访问更多数据来提升性能，即“更多经验”的机器代名词。机器通过深度学习获得足够经验后，即可用于特定的任务，如驾驶汽车、识别田地作物间的杂草、确诊疾病、检测机器故障等。

## TensorFlow 简单原理

tensorflow会将所有的属性进行包装, 同时也提供不同的transformer. 本文主要使用的就是tensorflow.

**Tensor**: tensorflow的数据, 名为张量. 指的是在tensorflow中运行的数据. 多维数据, tensorflow的基本思路就是针对张量进行操作(flow). Tensor可以为值/向量/矩阵/多维数据.
**operation**: tensorflow的运算节点. 所有的操作都是operation. 类似于算子
**图(graph)**: 整个程序的结构, 类似于storm的节点运算图结构, 在tensorflow1.0中使用的是静态图结构, 因此数据的导入导出需要使用到多个工具进行操作. 在tensorflow2.0中使用的是动态图结构, 更加灵活, 因此很多操作也就简化了. 只是2.0并不完全向下兼容.
**session**: tensorflow运算程序的图, 在2.0中基本弃用

tensorflow简单的操作

```python
import numpy as np
import tensorflow as tf

x = [[1.]]  # 定义一个1行一列的数组
m = tf.matmul(x, x)  # 得到tf封装的Tensor tf.Tensor([[1.]], shape=(1, 1), dtype=float32)
print(m)  # 由于使用的是tensorflow2.0, 动态图的因素, 可以直接打印架构

x = tf.constant([[1, 9], [3, 6]])
print(x)
# tf.Tensor(
# [[1 9]
#  [3 6]], shape=(2, 2), dtype=int32)

print(x.numpy())  # 也可以转换为numpy直接获取
# [[1 9]
#  [3 6]]

x = tf.cast(x, tf.float32)  # 类型转换
print(x)
# tf.Tensor(
# [[1. 9.]
#  [3. 6.]], shape=(2, 2), dtype=float32)

x_1 = np.ones([2, 2])  # 一个2x2的矩阵
x_2 = tf.multiply(x_1, 2)  # 简单的乘法操作
print(x_2)
# [[2. 2.]
#  [2. 2.]], shape=(2, 2), dtype=float64)

```

## 原理结构

深度学习的基本思路就是特征处理. 主要应用领域是计算机视觉和自然语言. 主要目的就是提取合适的特征.

特征工程的作用:

- 数据特征决定了模型的上限
- 预处理和特征提取是核心
- 算法与参数选择决定了如何逼近这个上限

常用的开源数据集

>**CIFAR数据集**
>CIFAR数据集是一个知名的图像识别数据集。CIFAR-10包含10个类别，50,000个训练图像，彩色图像大小：32x32，10,000个测试图像。CIFAR-100与CIFAR-10类似，包含100个类，每类有600张图片，其中500张用于训练，100张用于测试；这100个类分组成20个超类。图像类别均有明确标注。CIFAR对于图像分类算法测试来说是一个非常不错的中小规模数据集。
>
>**Open Image**
>Open Image是一个包含~900万张图像URL的数据集，里面的图片通过标签注释被分为6000多类。该数据集中的标签要比ImageNet（1000类）包含更真实生活的实体存在，它足够让我们从头开始训练深度神经网络。
>
> **Mnist**
> Mnist数据集:深度学习领域的“Hello World!”，入门必备！MNIST是一个手写数字数据库，它有60000个训练样本集和10000个测试样本集，每个样本图像的宽高为28*28。
>
> **ImageNet**
> ImageNet数据集:对深度学习的浪潮起了巨大的推动作用。深度学习领域大牛Hinton在2012年发表的论文《ImageNet Classification with Deep Convolutional Neural Networks》在计算机视觉领域带来了一场“革命”，此论文的工作正是基于Imagenet数据集。
> Imagenet数据集有1400多万幅图片，涵盖2万多个类别；其中有超过百万的图片有明确的类别标注和图像中物体位置的标注，具体信息如下：
>
> - Total number of non-empty synsets: 21841
> - Total number of images: 14,197,122  
> - Number of images with bounding box annotations: 1,034,908
> - Number of synsets with SIFT features: 1000
> - Number of images with SIFT features: 1.2 million
>
> **COCO**
> COCO(Common Objects in Context):是一个新的图像识别、分割和图像语义数据集。COCO数据集由微软赞助，其对于图像的标注信息不仅有类别、位置信息，还有对图像的语义文本描述，COCO数据集的开源使得近两三年来图像分割语义理解取得了巨大的进展，也几乎成为了图像语义理解算法性能评价的“标准”数据集。COCO数据集具有如下特点：
>
> - Object segmentation
> - Recognition in Context
> - Multiple objects per image
> - More than 300,000 images
> - More than 2 Million instances
> - 80 object categories
> - 5 captions per image
> - Keypoints on 100,000 people

## 神经网络基础

### 前向传播 (Forward propagation)

**损失函数(lose function)**, 损失函数是当得到一个结果以后才能进行评估的.

$$L_i = \sum_{j\neq{y_i}}max(0,s_j-{s_{y_i}}+1)$$

> 损失 = 错误类别 - 正确类别 + 1, 损失最低为0
> 将每一个损失相加获取当前分类的损失值

![d2](./images/d2.png)

比如我们存在3个类别, 数据得分会进行计算.比如我们计算第一张本来应该为cat, 但是错误类别为car, 因此我们用car的错误类别得分`5.1`减去cat的类别`3.2`然后加一个常量1, 获得car针对于cat的损失为`2.9`. 然后frag对cat的损失为frag的得分`-1.7`减去cat的`3.2`会得到一个负数, 因此不计算. 最终当前图片的损失值为`2.9`, 以此类推.

![d3](./images/d3.png)

*需要注意的是常量1是损失函数的delta容忍程度. 如果损失差距过小的话可能也会是错误的结果.*

> **正则化**
> 如果两个模型的损失函数值相同, 两个模型的效果未必一样. 尤其是出现过拟合的情况.
>
> $$L=\frac{1}{n}\sum_{i=1}^{N}\sum_{j\neq{y_i}}max(0,f(x_i;W)_j-f(x_i;W)_{y_i} + 1)+\lambda{R}(W)$$
>
> 其中的$\lambda{R}(W)$就是正则化的惩罚项. 通常情况下,正则化的惩罚项使用平方数$R(W)=\sum_{k}\sum_{l}W^2_{k,l}$处理就可以了
>
> *神经网络是一个极其强大的模型, 对于强大的模型, 存在的过拟合风险就越大.*

#### softmax 分类器

在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。

当我们获取了损失值以后我们希望可以将我们的数值转变成一个概率. 对于概率的比较就更加的直观并且更容易计算. 因此我我们可以引入sigmoid函数帮助我们, Sigmoid函数:

$${S(x)={\frac{1}{1+e^{-x}}}={\frac{e^{x}}{e^{x}+1}}=1-S(-x)}$$

假设我们对输入进行计算, 得出了如下的得分. 然后获得$e^x$也就是e的x次幂的映射, 这个结果就是放大差异(函数图像是指数图像). 最后再进行的分值占总体得分的百分比, 从而得到一个得分值的概率. 最后使用损失值方程进行计算得到我们想要的损失值: `0.89`

损失值: $L_i=-logP(Y=y_i|X=x_i)$

![d4](./images/d4.png)

> 综上所述, 我们可以得到归一化函数: $P(Y=k|X=x_i)={\frac{e^{s_{k}}}{\sum_{j}e^{s_{j}}}}{\text{ where }}s=f(x_i;W)$
> **也就是我们的前向传播的损失值计算公式**
> 通过损失值我们就可以使用梯度下降算法, 也就是反向传播

![d5](./images/d5.png)

### 反向传播(Back propagation)

通过前向传播得到了损失值, 就可以进行梯度下降的计算了, 但是首先需要看一下神经网络如何使用$W_i$, 也就是参数矩阵.

在神经网络中, 每一层的$\theta$(也就是上面的$w_i$)都会给当前的x值进行计算. 比如我们存在多层运算:

1. 第一层运算的时候我们给$x$进行运算, 得到了$(w_1x)$
2. 第二层运算的时候我们得到了$((w_1x)w_2)$
3. 之后运算都基于前一层计算结果$[w_1w_2x]w_3$
4. 知道所有的计算都被运算$[w_1w_2...w_{n-1}x]w_n$

梯度下降公式: $J(\theta_{0},\theta_{1})=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)$

#### 原理

当我们对一个数值进行包装性质的操作的时候, 我们可以反向计算出最后一次计算节点对我们最终结果的影响. 比如在下图中, 我们得到了最终结果`-12`, 我们可以尝试查看数值z对我们最终结果的影响有多大(通过求出z的偏导$\frac{\partial{f}}{\partial{z}}$).

而x和y的计算, 我们假设x+y的结果为q. 因此, 如果我们相对x求偏导, 我们可以计算出x的贡献: $\frac{\partial{f}}{\partial{q}}\cdot\frac{\partial{q}}{\partial{x}}$. 同理可以得到y的偏导.

![d6](./images/d6.png)

在神经网络中的反向传播, 我们是逐层进行计算的. 也被称为**链式法则**.

![d7](./images/d7.png)

>逐层的计算梯度
>
>![d8](./images/d8.png)
>
> 通过上图我们计算出来结果为0.73而返回的损失值为1. *绿色的数值是我们一步一步计算出来的数值. 红色的数值就是我们的梯度损失值.*
>
> 1. 最后一步的计算方程为$\frac{1}{x}$, 导数为$-\frac{1}{x^2}$, 那么结果就为$-\frac{1}{1.37}=-0.53$, 最后一步的梯度就为`-0.53`
> 2. $x+1$ 的偏导数为常数 $1$, 那么结果就是上一层梯度结果乘以本层导数, 就是 $-0.53 * 1 = -0.53$
> 3. $e^x$ 的偏导还是 $e^x$, 因此我们本层导数结果为$e^{-1}$, 本层的梯度结果就是$e^{-1} * -0.53 = 0.1949$, 这里向上取整了
> 4. 继续求偏导...
> 5. 当遇到多元素梯度, 以红框为例, 上一个梯度结果为`0.20`我们本层的运算为$x_0*w_0$的乘法运算. 对于$w_0$而言, 乘法的导数是取x的常量$x_0$也就是`-1.0`, 因此$-1.0*w$导数就是$-1.0$, $w_0$的梯度结果就是$-1.0 *0.20 = -0.20$. 对于$x_0$而言, 乘法的导数是取w的常量$w_0$也就是`2.0`, 因此$2.0*x$导数就是$2$, $x_0$的梯度结果就是$2.0 *0.20 = 0.4$.
>
> 从而计算出所有节点的全部结果. 在反向传播的时候我们可以对一个整体部分求偏导.
> ![d9](./images/d9.png)
>
> 实际计算的时候我们会对矩阵进行求偏导, 工具会完成计算.

反向传播门单元

- 加法门单元: 多元素均等
- 乘法门单元: 计算题都的时候直接将数值进行互换, 同时附加上一层梯度.
- max单元: 直接赋值当前梯度给max的输入

![d10](./images/d10.png)

### 模型架构

![d10](./images/d10.jpg)

层次性: 一层一层的数据进行计算
神经元: 节点, 第一层的输入就是特征
全连接: 每一个节点都连接到前一层或者后一层的节点. 每一条线代表的就是一个权重$w$. 因此输入的特征层和隐层1的箭头就是一个`3x4`的权重矩阵, 通过运算得到一个`1x4`的输出结果. 隐层2的输入就是隐层1的输出, 箭头就是一个$4x4$的权重矩阵, 然后进行计算.
非线性: 当每一层计算出结果来以后会进行一次非线性函数变换. sigmod函数就是一个非线性函数, 或者max函数, 这个函数会在每一次矩阵计算结束以后, 也就是神经元上计算. 结果也就是$x_n=W_n[sigmod(W_{n-1}*x_{n-1})]$

特点: 神经元越多, 过拟合可能性就越大. 速度也就会比较慢. <https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html>

#### 激活函数

当矩阵计算结束以后, 会进行非线性的激活函数, 常见的激活函数为sigmoid,Relu,Tanh等

sigmoid: $\frac{1}{1+e^{-x}}$, 计算准确性比较高, 但是近些年不常用了, 因为需要计算梯度的原因, sigmoid计算梯度的效率很低, 尤其是大数或者小数的时候, 此时可能会出现梯度消失的现象, 也就是梯度结果为0. 一旦出现梯度消失的情况, 后续层的梯度计算结果都为0.
Relu: $max(0,x)$, 用的比较多, **比较实用**, 梯度的计算很容易, 并且不存在梯度消失的情况.
Tanh: $\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$, sigmoid和tanh激活函数有共同的缺点：即在z很大或很小时，梯度几乎为零，因此使用梯度下降优化算法更新网络很慢。

#### 标准化, 初始化

不同的预处理结果会是的模型的效果发生很大的差异
![d11](./images/d11.png)

神经网络初始化的参数需要初始化, 通常我们都是用随机策略来进行参数初始化 `W=0.01*np.random.randn(D,H)`

#### drop-out

> dropout是一个七伤拳的问题, 主要处理过拟合
> 在训练的时候随机杀死神经元. 每一次迭代的时候都进行一次随机杀死神经元操作.

drop-out训练过程每层随机比例杀死. 在测试的时候, 直接使用整体神经网络进行测试. 主要目的就是为了防止模型过于复杂.

![d12](./images/d12.png)

## 神经网络代码

tensorflow2将大量使用Keras构建模型. <https://tensorflow.google.cn/api_docs/python/tf>

常用的部分参数

- activation：激活函数的选择，一般常用relu
- kernel_initializer, bias_initializer：权重与偏置参数的初始化方法，有时候不收敛换种初始化就突然好使了. 思考一下混沌理论, 无论多么细微的初始化差异会在大数据运算后呈现完全不同的结果. 所以如果出现不收敛的问题需要尝试不同的初始化方法来进行收敛.
- kernel_regularizer, bias_regularizer：要不要加入正则化，
- inputs：输入，可以自己指定，也可以让网络自动选
- units：神经元个数

> 在tensorflow1.0中, 如果需要搭建神经网络, 需要构建多个参数矩阵, 比如3x3x3x1的神经网络节点, 我们则需要2个3x3的矩阵和一个3x1的numpy矩阵,同时需要配置权重参数. 而在tensorflow2.0中, 不需要设置任何初始权重参数.

### 数据预测

```python
# 处理时间数据
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow import keras
import tensorflow as tf
# 不同版本的引用会不一样, 其他版本使用tensorflow.keras
# 也可以直接使用tf.keras.layers
from tensorflow.python.keras import layers

"""
搭建神经网络进行气温预测
"""
# 导入数据, actual就是数据
#
#    year  month  day  week  temp_2  temp_1  average  actual  friend
# 0  2016      1    1   Fri      45      44       44      45      24
# 1  2016      1    2   Sat      46      45       44      45      24
# 2  2016      1    3   Sun      47      46       44      45      24
# 3  2016      1    4   Mon      48      47       44      45      24
# 4  2016      1    5  Tues      49      48       44      45      24
features = pd.read_csv('E:\\Workspace\\ml\\code-ml\\ml\\python\\csv\\temps.csv')
print(features.head())

years = features['year']
month = features['month']
day = features['day']
# zip方法: 将多个list压制为元祖: a=[a,b,c]; b=[1,2,3]; zip(a,b)=[(a,1),(b,2),(c,3)]
# 通过for循环list, 然后返回值构建新的list, 返回的元祖赋值到(year,month,day)中
# 通过string创建返回值"2016-1-2"等等的列表
dates = [str(int(years)) + '-' + str(int(month)) + '-' + str(int(day)) for years, month, day in zip(years, month, day)];
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

################################# 数据可视化展示 #############################

# matplotlib 也提供了几种我们可以直接来用的内建样式
# 导入 matplotlib.style 模块 - 探索 matplotlib.style.available 的内容，里面包括了所有可用的内建样式
# 这些样式可以帮我们改变背景颜色, 改变网格, 消除毛刺等等. 这里我们制定了fivethirtyeight的样式
plt.style.use('fivethirtyeight')
# 设置布局, 获取四格图像的subplot, 2x2的图像矩阵
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)
# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''), ax1.set_ylabel('Temperature'), ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''), ax2.set_ylabel('Temperature'), ax2.set_title('Previous Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'), ax3.set_ylabel('Temperature'), ax3.set_title('Two Days Prior Max Temp')
# 朋友预测
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'), ax4.set_ylabel('Temperature'), ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show() # 展示一下

################################### 建模 ###############################
# 1. 特征工程
# 将数据进行特征修改, 将weekday改为数字. 有多种处理方式, 这里使用的是独热处理
features = pd.get_dummies(features)
print(features.head(5))
#    year  month  day  temp_2  temp_1  average  actual  friend  week_Fri  week_Mon  week_Sat  week_Sun  week_Thurs  week_Tues  week_Wed
# 0  2016      1    1      45      45     45.6      45      29         1         0         0         0           0          0         0
# 1  2016      1    2      44      45     45.7      44      61         0         0         1         0           0          0         0
# 2  2016      1    3      45      44     45.8      41      56         0         0         0         1           0          0         0
# 3  2016      1    4      44      41     45.9      40      53         0         1         0         0           0          0         0
# 4  2016      1    5      41      40     46.0      44      41         0         0         0         0           0          1         0

# 处理x和y
labels = np.array(features['actual'])  # y值
features = features.drop('actual', axis=1)  # 删除actual column, drop()方法默认删除行, axis表示这里删除的是列
# 获取column的名称单独保存到另一个参数中.
# columns参数获取全部的column名称, 同样可以使用index和values获取需要的数据, index是行坐标, values是全部数据
feature_list = list(features.columns)
features = np.array(features)  # 将feature的DataFrame类型转换为ndArray类型

# 对数据进行预处理, 还是之前sklearn的standardscaler
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features)
# [[ 0.         -1.5678393  -1.65682171 ... -0.40482045 -0.41913682 -0.40482045]
#  [ 0.         -1.5678393  -1.54267126 ... -0.40482045 -0.41913682 -0.40482045]
#  [ 0.         -1.5678393  -1.4285208  ... -0.40482045 -0.41913682 -0.40482045]
#  ...

# 2. 构建模型
# 这里我们使用的是dense mode全连接层, 其中也包含卷积层cropping等等
# <https://tensorflow.google.cn/api_docs/python/tf/keras>
# tf.keras.layers.Dense(
#     units,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )
model = tf.keras.Sequential()  # 创建模型
model.add(layers.Dense(16))  # 第一层16个神经元, 也就是16个特征
model.add(layers.Dense(32))  # 第二层32个神经元
model.add(layers.Dense(1))  # 输出单元1

# 初始化网络模型
# model.compile(optimizer='sgd', loss='mse')
# optimizer 优化迭代器. 损失函数的迭代器. 这里可以使用上面的简写, adam也是常用的迭代器.
# loss 损失函数, mse是非常常见的损失函数, 不同损失函数对最终结果影响很大.
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')

# 训练, tensorflow1.0中需要创建session等等, 在tensorflow2.0中只需要基础属性就可以了
# validation_split 就是测试集分解
# epochs 运算迭代的次数, 这里制定了迭代10次
# batch_size 每一次优化器迭代多少个样本, 这里选择的是64个样本, 越大越好64/128/256
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 50.4199 - val_loss: 724.1822
# 可以看到在训练集中的损失值为724.182, 但是训练集的损失为50.4199, 说明模型出现了过拟合的状态.

model.summary()
# Model: "sequential"
# -----------------------------------------------------------------
#  Layer (type)                Output Shape              Param #
# =================================================================
#  module_wrapper (ModuleWrapp  (None, 16)               240  er)  一共14个特征, 第一层有16个神经元, 总共16x14=224, 再加上偏置函数16为240
#  module_wrapper_1 (ModuleWra  (None, 32)               544  pper) 32*16=512, 再加上32个偏置函数总共544
#  module_wrapper_2 (ModuleWra  (None, 1)                33  pper) 32 * 1 + 1 = 33
# =================================================================
# Total params: 817
# Trainable params: 817
# Non-trainable params: 0


# 3. 重新构建模型, 尝试不同的结果
# 使用random_normal初始化函数, 高斯分布
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal'))
model.add(layers.Dense(32, kernel_initializer='random_normal'))
model.add(layers.Dense(1, kernel_initializer='random_normal'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)
# 好一丢丢, 但是复杂的数据可能会出现不同的结果
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 53.7003 - val_loss: 692.4907

# 使用正则化修正, L2的正则化, lambda的值设置为0.03
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=100, batch_size=64)
# 正则化会非常稳定
# Epoch 10/10
# 5/5 [==============================] - 0s 6ms/step - loss: 37.9034 - val_loss: 583.6664
# Epoch 100/100
# 5/5 [==============================] - 0s 6ms/step - loss: 52.9017 - val_loss: 31.6005

##########################################结果预测####################################
# 放入测试集
predict = model.predict(input_features)  # 这里不应该把预测结果直接放入, 因为测试集会完美符合, 但是, 时间有限只能上车了
print(predict.shape)
# 转换日期
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, month, day)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60')
plt.legend()

# 图名
plt.xlabel('Date'), plt.ylabel('Maximum Temperature (F)'), plt.title('Actual and Predicted Values')
plt.show()
```

数据的分布
![d13](./images/d13.png)

![d14](./images/d14.png)

### 分类

使用手写输出的分类模型, 让他可以给出一个分类的结果

```python
# 处理时间数据
import gzip
from pathlib import Path
import pickle

from matplotlib import pyplot
import requests
import tensorflow as tf
# 不同版本的引用会不一样, 其他版本使用tensorflow.keras
# 也可以直接使用tf.keras.layers
from tensorflow.python.keras import layers

"""
分类任务
"""
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
print(PATH)  # data\mnist

PATH.mkdir(parents=True, exist_ok=True)  # 创建文件夹
URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/'
FILENAME = 'mnist.pkl.gz'

# url下载到本地
if not (PATH / FILENAME).exists():
    print("downloading...")
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# 解压缩, 然后抽取train和test
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    print("unzipping...")
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)))
pyplot.show()
print(x_train.shape)  # 展示一波
print(y_train[0])  # 5, 这里的y不是onehot类型,属于单一输出

############################ 模型构建 ############################
# 数据是一个28*28的图片, 得到的784个pixel
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 分为10类

# 由于输入的结果值y属于属性方法, 而不是onehot类型, 我们可以使用不同的损失函数
# 比如 CategoricalCrossentropy 需要一个onehot形式 (we expect labels to be provided in a one_hot representation.)
# 损失函数可能会变得非常离谱, 所以如果损失值出现问题, 检查损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])  # 展示准确率, 准确率的计算metric
model.fit(x_train, y_train,
          validation_split=0.25, epochs=5, batch_size=64,
          validation_data=(x_valid, y_valid))
```

### 数据处理

tensorflow中数据大部分为矢量数据, 设计初衷也是使用矢量数据作为数据源的.

TensorFlow 1.X中的常见使用模式是“水槽”策略，其中所有可能的计算的合集被预先排列，然后通过 session.run() 评估选择的张量。在TensorFlow 2.0中，用户应将其代码重构为较小的函数，这些函数根据需要调用。通常，没有必要用 tf.function 来修饰这些较小的函数，仅使用 tf.function 来修饰高级计算 - 例如，训练的一个步骤或模型的正向传递。

```python
import gzip
from pathlib import Path
import pickle

import numpy as np
import requests
import tensorflow as tf
from tensorflow.python.keras import layers

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
print(PATH)  # data\mnist

PATH.mkdir(parents=True, exist_ok=True)  # 创建文件夹
URL = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/'
FILENAME = 'mnist.pkl.gz'

if not (PATH / FILENAME).exists():
    print("downloading...")
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    print("unzipping...")
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

"""
数据集, 以及常用函数
"""
input_data = np.arange(16)
print(input_data)  # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]

# 直接将ndarray类型的数据传入tensorflow, 得到的flow数据是可以遍历的
dataset = tf.data.Dataset.from_tensor_slices(input_data)
for data in dataset:
    print(data)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # tf.Tensor(2, shape=(), dtype=int32)
    # ...

# repeat操作
# 将dataset中的数据repeat一份, 同样序列
dataset = tf.data.Dataset.from_tensor_slices(input_data)
dataset = dataset.repeat(2)
for data in dataset:
    print(data)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # ...
    # tf.Tensor(15, shape=(), dtype=int32)
    # tf.Tensor(0, shape=(), dtype=int32)
    # tf.Tensor(1, shape=(), dtype=int32)
    # ...
    # tf.Tensor(15, shape=(), dtype=int32)

# batch 操作, 批量操作, 将数据分批打包
dataset = tf.data.Dataset.from_tensor_slices(input_data)
dataset = dataset.batch(4)  # 每4个组成一个batch
for data in dataset:
    print(data)
    # tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
    # tf.Tensor([4 5 6 7], shape=(4,), dtype=int32)
    # tf.Tensor([ 8  9 10 11], shape=(4,), dtype=int32)
    # tf.Tensor([12 13 14 15], shape=(4,), dtype=int32)

# shuffle操作, 打乱顺序
# buffer_size 缓存区, 随机构建的时候使用缓存区进行抽取, 然后从缓存区中随机抽取构建序列. 类似于随机滑动窗口.
# 比如buffer_size为10, 那么就将1-10放入缓存区, 然后随机抽取, 抽取一个以后将11放入随机抽取.
# 此时, 11绝对不会出现在第一个位置, 而第一个位置也必然只能是1-10中的一个
# 因此, 如果buffer_size为1的时候就是不进行乱序操作, 而buffer_size为数据长度相同时就是全局随机排列.
dataset = tf.data.Dataset.from_tensor_slices(input_data).shuffle(buffer_size=10).batch(4)
for data in dataset:
    print(data)
    # tf.Tensor([ 1  0 11  6], shape=(4,), dtype=int32)
    # tf.Tensor([13  5 15  2], shape=(4,), dtype=int32)
    # tf.Tensor([ 8  7  3 14], shape=(4,), dtype=int32)
    # tf.Tensor([10  9  4 12], shape=(4,), dtype=int32)

# 基于数据集的重新训练
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).repeat()
valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32).repeat()
model.fit(train, epochs=5, steps_per_epoch=100, validation_data=valid, validation_steps=100)
```

### 案例

```python
import matplotlib.pyplot as plt
from tensorflow import keras

# 下载服装数据集
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 输出类别,我们输出的类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)  # (60000, 28, 28)
print(len(train_labels))  # 60000
print(test_images.shape)  # (10000, 28, 28)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()  # 添加颜色热度
plt.grid(False)  # 关闭网格
# plt.show()

#### 处理图像, 需要注意的是, 预处理需要从测试集和预测同时使用
#train_images = train_images / 255.0
#test_images = test_images / 255.0

# 展示图像
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()

## 训练模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # 将图像进行拉伸, 成为784的feature, 同时定义输入层
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
```

### 评估

```python
####评估模型
# 指定测试集的x和测试集的y就可以了, 直接出结果
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)  # Test accuracy: 0.8847000002861023

predictions = model.predict(test_images)
print(predictions.shape)  # (10000, 10)
print(predictions[0])
# [7.6147963e-08 5.5699688e-11 2.4855829e-10 6.9418427e-09 4.1508761e-09
#  2.9993954e-04 1.7000576e-07 5.8553129e-02 2.8515702e-08 9.4114667e-01]

print(np.argmax(predictions[0]))  # 9
```

### 模型的保存和使用

保存模型的时候会保存权重参数和网络的结构

```python
### 模型的保存
model.save('fashion_model.h5')


# 将网络模型输入到文件中
config_json = model.to_json()
# '{"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}'
with open('config.json', 'w') as json:
    json.write(config_json)

# 获取权重参数
weights = model.get_weights()
print(weights)
# 也可以存入文件
model.save_weights("fashion.weights")
```

读取模型:

```python
from tensorflow import keras

# 读取模型
model = keras.model.load_model('fashion_model.h5')
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
predictions = model.predict(test_images)
print(predictions.shape)  # (10000, 10)


# 读取json模型
model = keras.models.model_from_json(config_json) # config_json是json字符串
model.summary()

# 读取权重
model.load_weights('feshion.weights')
```

## 卷积神经网络 Convolutional Neural Networks

卷积神经网络 – CNN 最擅长的就是图片的处理。它受到人类视觉神经系统的启发。目前 CNN 已经得到了广泛的应用，比如：人脸识别、自动驾驶、美图秀秀、安防等很多领域。

CNN 有2大特点：

- 能够有效的将大数据量的图片降维成小数据量
- 能够有效的保留图片特征，符合图片处理的原则
