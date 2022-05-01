# 深度学习

深度学习是机器学习的一个分支。 许多传统机器学习算法学习能力有限，数据量的增加并不能持续增加学到的知识总量，而深度学习系统可以通过访问更多数据来提升性能，即“更多经验”的机器代名词。机器通过深度学习获得足够经验后，即可用于特定的任务，如驾驶汽车、识别田地作物间的杂草、确诊疾病、检测机器故障等。

## TensorFlow

tensorflow会将所有的属性进行包装, 同时也提供不同的transformer.

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

深度学习的基本思路就是特征处理. 主要应用领域是计算机视觉和自然语言.
