# Machine learning

## anaconda的基本使用

机器学习使用的是 anaconda, 需要安装 anaconda. 然后配置环境变量.

添加环境变量

```text
D:\anaconda
D:\anaconda\Scripts
D:\anaconda\Library\bin
```

如果需要使用 pyspark 的话需要制定`SPARK_HOME`环境变量. 安装 py4j. (这里的 pip 命令在 anaconda 的 script 目录中). 可以控制版本使用==

```text
E:\Workspace\ml>pip install py4j
E:\Workspace\ml>pip install pyspark==3.2.1
```

然后配置 anaconda 的环境到 intellij 的编译器中

配置一个简单的是 park 项目:

```python
from pyspark import SparkConf, SparkContext

# main 方法
if __name__ == '__main__':
    conf = SparkConf()
    conf.setMaster("local")
    conf.setAppName("test")
    sc = SparkContext(conf=conf)
    lines = sc.textFile("./words")
    words = lines.flatMap(lambda line: line.split(" "))
    # words.foreach(print)
    pairWords = words.map(lambda word: (word, 1))  # 类似于(word)=>{return (word,1)}
    reduce_result = pairWords.reduceByKey(lambda v1, v2: v1 + v2)  # 类似于(v1,v2)=>{return v1+v2}
    # result.foreach(print)
    # 排序
    result = reduce_result.sortBy(lambda tp: tp[1])
    result.foreach(print)
```

## 算法

[toc]

### 线性回归

**回归:** 根据大量的样本点之间的关系, 反推与点的关系最拟合的那一条线. 对于直线的$y=w_0+w_1x$中. x为`特征`或者`维度` w_1为`斜率`, w_0为`截距`, y成为`真实值`或者`预测值`

目的: 根据样本点来计算出$w_0$和$w_1$的值. 也就确定了模型.

#### lose function

**线性回归的误差公式/最小二乘法公式/损失函数(lose function)**
$$error = \frac{1}{2m}\sum^{m}_{i=1}(y_i-h_\theta(x_i))^2$$

**各个数字的意义**:
`m` : 一共有m个样本点
$y_i$ : 第i个样本点, 也就是真实的样本点的y值
$h_\theta(x_i)$ : 计算出来的回归方程的y值 也就是回归的预测值. 此处的$\theta$

- 此处的$x_i$就是第i个样本$x$
- 可以看做为 $\theta^t*x$
- $\theta$ 就是处于行列式的$(w_0,w_1,w_2...)$的常量, 因此$\theta^t*x$ 就是$w_0+w_1x_1+w_2x_2...$.

整体的误差`error`就是: 把每一个样本点$y$与我们计算出来的公式的方程的结果的$y$相减, 得出来的每一个样本平方. 然后将所有的$m$个样本点相加, 之后除以$2m$.

如果需要获得lose function的最小值. 我们只需要对应最小值的$\theta$就可以了. 因此可以对error进行求导. 找出导数为0的时候才能获得数值. *注意*: 损失函数是相对于$h_{\theta}$的一个二次方程, 因此存在最小值.

#### 梯度下降递归

对于error求最小值, 我们无法知道未知的$\theta$, 没办法根据一个确定的一组$w$. 我们可以使用梯度下降的方法, 找到最小值.

$${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)$$

这里的$\partial$表示的是求导, 我们根据导数求出梯度下降递推公式

$$w_j := w_j - \alpha\frac{1}{m}\sum^{m}_{i=1}(\sum^{n}_{j=0}w_jx^{(i)}_j-y^{(i)})x^{(i)}_j$$

其中，`:=`为赋值的含义, $\alpha$ 为学习速率也就是步长, 可以理解为我们下山时每走一步的距离

**迭代停止条件**: 当两次递归误差非常小的时候, 当两次迭代到达一定数值的时候.

拟合状态
过拟合: 在训练集中表现很好, 在测试集中表现不好
欠拟合: 不能很好地适应我们的训练集

**随机梯度下降优点**:

- 随机梯度下降的"随机”体现在进行梯度计算的样本是随机抽取的n个,与直接采用全部样本相比,这样*计算量更少*
- 随机梯度下降善于解决大量训练样本的情况
- 学习率决定了梯度下降的速度,同时,在SGD的基础上引入了”动量”的概念，从而进一步加速收敛速度的优化算法也陆续被提出

#### 代码实现

#### scala 代码实现

spark存在老版本核心版本的配置不同,老版本中,可以使用linearRegressionWithSGD方法来进行训练模型. 但是在新版本中被弃用

##### scala 2.x 版本 (过期)

```scala
package com.snowave.machine.learning.linearRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LinearRegression {
  def main(args: Array[String]): Unit = {
    // 创建spark对象
    val conf = new SparkConf().setAppName("LinearRegressionWithSGD").setMaster("local");
    val sparkContext = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN);

    // 读取样本
    val data: RDD[String] = sparkContext.textFile("data/lpsa.data")
    // 获取labeledPoint, 一个对象封装一个label和一堆feature
    val examples: RDD[LabeledPoint] = data.map { line =>
      val parts = line.split(',')
      val y = parts(0)
      val xs = parts(1)
      // 把每一行创建一个labelpoint, point存在一个label和一堆features向量, 这里的label为y, features为x
      LabeledPoint(y.toDouble, Vectors.dense(xs.split(' ').map(_.toDouble)))
    }
    // 创建测试组, 随机将rdd中切分成两个
    val train2TestData: Array[RDD[LabeledPoint]] = examples.randomSplit(Array(0.8, 0.2), 1L);

    /**
     * 迭代操作
     * 训练一个多元线性回归模型收敛（停止迭代）条件：
     * 1、error值小于用户指定的error值
     * 2、达到一定的迭代次数
     */
    val numIterations = 1000 // 定义最大迭代次数

    //在每次迭代的过程中 梯度下降算法的下降步长大小    0.1 0.2 0.3 0.4  // 这里0.1 最好
    val stepSize = 0.1
    val miniBatchFraction = 1

    val lrs = new LinearRegressionWithSGD();
    lrs.setIntercept(false) // 设置当前训练的模型有截距(常量)
    lrs.optimizer.setStepSize(stepSize) // 设置步长, 可以从0.0001开始设置
    lrs.optimizer.setMiniBatchFraction(miniBatchFraction) // 每次计算后拿出多少来进行error的迭代计算, 这里指定为一批次为1个

    /**
     * 开始用训练集训练数据
     */
    val model = lrs.run(train2TestData(0))

    println("weight = " + model.weights) // 有8个参数, 那么就有8个权重
    println("intercept = " + model.intercept) // 截距, 有一个

    // 对样本进行测试. 获取测试用例中的样本的features, 然后对于结果进行测试
    val prediction = model.predict(train2TestData(1).map(_.features))
    val predictionAndLabel: RDD[(Double, Double)] = prediction.zip(train2TestData(1).map(_.label)) // zip: rdd1和rdd2合成kv格式的rdd. 这里是获取一个tuple2的数据, key为测试结果,value为测试应当的结果

    val print_prediction = predictionAndLabel.take(20);

    for (i <- print_prediction.indices) { // 等同于 for (i <- 0 to print_prediction.length - 1) {
      println(print_prediction(i)._1 + "\t" + print_prediction(i)._2)
    }

    // 计算测试集平均误差
    val loss = predictionAndLabel.map {
      case (p, v) =>
        val err = p - v
        Math.abs(err)
    }.reduce(_ + _)
    val error = loss / train2TestData(1).count
    println("Test RMSE = " + error)
    // 模型保存
    //    val ModelPath = "model"
    //    model.save(sc, ModelPath)
    //    val sameModel = LinearRegressionModel.load(sc, ModelPath)
    sparkContext.stop()

  }
}
```

##### scala 3.x 版本

新版本中的scala使用的是linearRegression类.可以直接调用. 原始的GD类转换为了 streaming, 主要负责处理streaming类型的数据. 也可以进行简单的转换来处理

```scala
package com.snowave.machine.learning.linearRegression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, StreamingLinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LinearRegression {
  def main(args: Array[String]): Unit = {
    // 创建spark对象
    val conf = new SparkConf().setAppName("LinearRegressionWithSGD").setMaster("local");
    val sparkContext = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN);

    // 读取样本
    val data: RDD[String] = sparkContext.textFile("data/lpsa.data")
    // 获取labeledPoint, 一个对象封装一个label和一堆feature
    val examples: RDD[LabeledPoint] = data.map { line =>
      val parts = line.split(',')
      val y = parts(0)
      val xs = parts(1)
      // 把每一行创建一个labelpoint, point存在一个label和一堆features向量, 这里的label为y, features为x
      LabeledPoint(y.toDouble, Vectors.dense(xs.split(' ').map(_.toDouble)))
    }
    // 创建测试组, 随机将rdd中切分成两个
    val train2TestData: Array[RDD[LabeledPoint]] = examples.randomSplit(Array(0.8, 0.2), 1L);

    /**
     * 迭代操作
     * 训练一个多元线性回归模型收敛（停止迭代）条件：
     * 1、error值小于用户指定的error值
     * 2、达到一定的迭代次数
     */
    val numIterations = 1000 // 定义最大迭代次数

    //在每次迭代的过程中 梯度下降算法的下降步长大小    0.1 0.2 0.3 0.4  // 这里0.1 最好
    val stepSize = 0.1
    val miniBatchFraction = 1

    val lrs = new StreamingLinearRegressionWithSGD()
    lrs.setNumIterations(numIterations)
    lrs.setStepSize(stepSize)
    lrs.setMiniBatchFraction(miniBatchFraction)
    lrs.algorithm.setIntercept(true)

    /**
     * 开始用训练集训练数据
     */
    val model = lrs.algorithm.run(train2TestData(0));

    println("weight = " + model.weights) // 有8个参数, 那么就有8个权重
    println("intercept = " + model.intercept) // 截距, 有一个

    // 对样本进行测试. 获取测试用例中的样本的features, 然后对于结果进行测试
    val prediction = model.predict(train2TestData(1).map(_.features))
    val predictionAndLabel: RDD[(Double, Double)] = prediction.zip(train2TestData(1).map(_.label)) // zip: rdd1和rdd2合成kv格式的rdd. 这里是获取一个tuple2的数据, key为测试结果,value为测试应当的结果

    val print_prediction = predictionAndLabel.take(20);

    for (i <- print_prediction.indices) { // 等同于 for (i <- 0 to print_prediction.length - 1) {
      println(print_prediction(i)._1 + "\t" + print_prediction(i)._2)
    }

    // 计算测试集平均误差
    val loss = predictionAndLabel.map {
      case (p, v) =>
        val err = p - v
        Math.abs(err)
    }.reduce(_ + _)
    val error = loss / train2TestData(1).count
    println("Test RMSE = " + error)
    // 模型保存
    //    val ModelPath = "model"
    //    model.save(sc, ModelPath)
    //    val sameModel = LinearRegressionModel.load(sc, ModelPath)
    sparkContext.stop()

  }
}
```

### 贝叶斯算法

贝叶斯分类算法是统计学的一种分类方法.

它是一类利用概率统计知识进行分类的算法。在许多场合，朴素贝叶斯(Naïve Bayes，NB)分类算法可以与决策树和神经网络分类算法相媲美，该算法能运用到大型数据库中，而且方法简单、分类准确率高、速度快。

$$P(B|A) = \frac{P(A|B) * P(B)}{P(A)} = P(A \cap B)$$

A条件下发生B的概率 = ($P_B$条件下发生$P_A$的概率 * $P_B$发生的概率) / $P_A$发生的概率
