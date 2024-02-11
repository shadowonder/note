# Note

Unit 就是void 返回值

```kotlin
// 返回值就是一个Any因为返回有的时候是int有的时候是String
when (number) {
    1 -> "test1"
    2 -> "test2"
    3 -> "test3"
    else -> -1
}
```

When 函数:

```kotlin
when (num) {
    in 0..59 -> println("F")
    in 60..79 -> println("C")
    in 80..89 -> println("B")
    in 90..100 -> println("A")
    else -> TODO("Not exists") // Nothing 类别,当前程序抛出异常, 类似于NotImplementedError异常
    // Exception in thread "main" kotlin.NotImplementedError: An operation is not implemented: Not exists
}
```

22 反引号

```kotlin
fun main() {
    `This is a test function`(5555)
}

// 在kotlin中In和is都是关键字, 因此使用反引号.
// 在kotlin中可以调用java的类, 但是在java中的某些方法名却是kotlin的关键字.
// 或者用于代码的反编译防御
fun `This is a test function`(num: Int) {
    println("Test")
}
```

## Kotlin函数

最简单的函数

```kotlin
// 调用 testFunction() 就直接返回test
fun testFunction() = "test"
fun testFunction1(v: String?) = if (v == null) "?" else "!" // 根据是不是null返回符号
```

### 匿名函数

```kotlin
/**
 * 匿名函数
 */
fun main() {
    println("This is atest".count())// 直接使用 返回13

    // it就是每一个字符. 重写了count方法, 如果是满足, 那么就计算为count
    // 使用一个匿名函数代替了count方法
    val len = "This is a test".count() {
        it == 't'
    }
    println(len) // 返回2
}
```

### 函数式定义, 匿名函数

```kotlin
/**
 * 隐式函数, 匿名函数, 函数式定义
 */
fun main() {
    println("This is atest".count())// 直接使用 返回13

    // 声明一个函数, 类似于typescript中的type, 或者接口
    val methodAction: () -> String

    // 实现体, 类似于java的声明和赋值
    methodAction = {
        // 最后一行就是返回值
        "test"
    }
    println(methodAction())
}

/**
 * 函数参数
 */
fun main() {
    val methodAction: (Int, Int, Int) -> String = { a, b, c ->
        // 最后一行就是返回值
        "test $a $b $c"
    }
    println(methodAction(1, 2, 3))
}

/**
 * 匿名函数, 就是一个大括号定义
 */
fun main() {
    // 类型推断为double
    val method1 = { v1: Double, v2: Float, v3: Int ->
        v1 + v2 + v3
    }
    println(method1(1.1, 1.1f, 1))
    println(method1(1.1, 1.1f, 1).javaClass.name) // Double
    val method2 = {
        1234
    } // 等同于 () -> Int
    println(method2())
}

/**
 * 函数参数
 */
fun main() {
    // 最后一个参数是一个函数, 于是我们可以直接实现这个函数.
    val result = logicApi("test", "passwd") { msg: String, code: Int ->
        println(msg)
    }
    println(result)
    // 如果不是最后一个就需要写在里面.
    logicApi2("test", "passwd", { msg: String, code: Int ->
        println(msg)
    }, { msg: String, code: Int ->
        println(msg)
    })
}

fun logicApi(username: String, password: String, result: (String, Int) -> Unit): String {
    result(username, 1);
    return "$username $password "
}

fun logicApi2(
    username: String,
    password: String,
    result: (String, Int) -> Unit,
    result2: (String, Int) -> Unit
): String {
    result(username, 1);
    return "$username $password "
}


/**
 * 将函数作为函数式参数调用
 */
fun main() {
    logicApi("test", ::call)
    // 适用对象也可以接受
    val obj = ::call
    logicApi("test", obj)
}

fun call(msg: String, code: Int) {
    println(msg)
    println(code)
}

inline fun logicApi(username: String, result: (String, Int) -> Unit) {
    result(username, 1)
}

/**
 * 函数作为返回值
 */
fun main() {
    val res = logicApi("test")
    res("a", 1)
}

fun logicApi(username: String): (String, Int) -> Unit {
    return { msg: String, code: Int ->
        println(msg)
        println(code)
    }
}
```

### 内联, inline

Kotlin 的 Lambda 为了完全兼容到 Java6，不仅增大了编译代码的体积，也带来了额外的运行时开销。为了解决这个问题，Kotlin
提供了`inline` 关键字。

从 Java8 开始，Java 借助 `invokedynamic`来完成的 Lambda 的优化。`invokedynamic`
用于支持动态语言调用。在首次调用时，它会生成一个调用点，并绑定该调用点对应的方法句柄。后续调用时，直接运行该调用点对应的方法句柄即可。
说直白一点，第一次调用`invokeddynamic`时，会找到此处应该运行的方法并绑定， 后续运行时就直接告诉你这里应该执行哪个方法。

> 函数参数如果包含一个函数的话, 那么就需要使用到内联
> 如果不使用内联的话, 例如上面的函数参数就会使用到函数的INSTANCE对其进行配置, 同时使用的就是一个null的模式
> 如果使用内联的话, 就会帮助我们生成类的对象模式
> ```java
> public final class ApplicationKt {
>   public static final void main() {
>     logicApi("test", "passwd", (Function2)null.INSTANCE);
>   }
>
>   // $FF: synthetic method
>   public static void main(String[] var0) {
>     main();
>   }
>
>   public static final void logicApi(@NotNull String username, @NotNull String password, @NotNull Function2 result) {
>     Intrinsics.checkNotNullParameter(username, "username");
>     Intrinsics.checkNotNullParameter(password, "password");
>     Intrinsics.checkNotNullParameter(result, "result");
>     result.invoke(username, 1);
>     result.invoke(password, 2);
>   }
> }
> ```
> 这样会造成性能的损耗.

内联Inline. 如果函数参数中有函数, 尽量使用Inline. JVM会自动进行优化.

```kotlin
/**
 * 内联函数
 * 如果函数不使用内联, 在调用的时候会生成多个对象来调用, 会造成性能的损耗
 */
fun main() {
    logicApi("test", "passwd") { msg: String, code: Int ->
        println(msg)
    }
}

// 使用lamba作为参数的函数就需要使用内联函数
// 使用了inline以后, 内存指针也就指向那个原始的对象.
// 如果使用内联, 就相当于C++的#define, 也就是宏定义, 或者宏替换, 也就是编译的时候将代码直接放入到调用处. 因此调用处没有任何函数或者对象开辟.
inline fun logicApi(username: String, password: String, result: (String, Int) -> Unit) {
    result(username, 1);
    result(password, 2);
}
```

> 还有一种场景，我是 API 的设计者，我不想 API 使用者进行非局部返回 ，改变我的代码流程。同时我又想使用 inline
> ，这样其实是冲突的。前面介绍过，内联会让 Lambda 允许非局部返回。
>
> crossinline 就是为了解决这一冲突而生。它可以在保持内联的情况下，禁止 lambda 从外层函数直接返回。
> ```kotlin
> inline fun runCatch(crossinline block: () -> Unit) {
> }
> ```

同时, 对指定的 Lambda 参数使用 noinline ，可以避免该 Lambda 被内联。

## 处理空属性

kotlin中使用逻辑方法消除了空指针异常.

```kotlin

/**
 * 语言的可空性特点
 */
fun main() {
//    nonNull()
//    safeLoad()
//    letSafeLoad()
//    assertForNull()
//    ifNullHandling()
}

fun nonNull() {
    // kotlin中不能使用null来进行赋值 name=null -> 报错
    var name: String = "test"
    println(name)

    // 声明时指定为可空类型
    var name2: String?
    name2 = null;
    println(name2);
}

/**
 * 安全调用操作符 "?"
 */
fun safeLoad() {
    var name: String?
    name = null

    // 如果name是可空类型, 如果哦想要使用name,必须给出null的解决方案.
    var capitalized = name?.capitalize() // 如果真是null, 后面的代码不执行.
    print(capitalized) // 输出null
}

/**
 * let 的安全调用
 */
fun letSafeLoad() {
    var name: String? = null
    name = ""

    val letResult = name?.let {
        // 如果进入这个block, 那么it必然不为Null
        if (it.isBlank()) {
            "None"
        } else {
            "[$it]"
        }
    }
    println(letResult)
}

/**
 * 对于空的断言操作
 */
fun assertForNull() {
    var name: String? = null

    // 不管name是不是null都执行, 就和java一样了. 断言操作
    val capitalize = name!!.capitalize() // 抛出空指针异常
    println(capitalize)
}


/**
 * If 对于空的处理
 */
fun ifNullHandling() {
    var name: String? = null

    if (name != null) {
        val r = name.capitalize()
        println(r)
    } else {
        println("name is null")
    }
}

/**
 * 空合并操作符
 */
fun mergeNull() {
    var name: String? = null
    println(name ?: "Name is null!")

    // 使用let进行合并操作符
    // 先用let执行一波, 然后在用null执行一波, 类似三元运算, 但是是两个独立运算体
    println(name?.let { "[$it]" } ?: "name is null!")
}
```

独立抛出异常

```kotlin
/**
 * 独立抛出异常
 */
fun main() {
    try {
        var info: String? = null;

        checkException(info) // 检查异常

        println(info!!.length)
    } catch (e: Exception) {
        e.printStackTrace()
    }
}

// 检测并抛出独立异常
fun checkException(info: String?) {
    info ?: throw CustomizeException()
}

// 自定义异常
class CustomizeException : IllegalArgumentException("Customized Exception")
```

```kotlin
/**
 * 使用先决条件函数处理异常
 */
fun main() {
    try {
        var info: String? = null;

        checkNotNull(info) // 检查异常
//        requireNotNull(info) // 不同的异常级别

        println(info!!.length)

        // 顺带一提, require函数可以对一个数据进行检测
        val test: Boolean = false
        require(test) // 先决条件处理, 如果不对, 抛出异常
        // require(boolean, ()=><T>) // 如果不满足boolean, 那么执行lambda
    } catch (e: Exception) {
        e.printStackTrace()
    }
}
```

### String 的处理

```kotlin
/**
 * String 的处理
 */
const val INFO = "This is a test String"

fun main() {
    // 1. substring
    println(INFO.substring(0, INFO.indexOf("i")))
    println(INFO.substring(0 until INFO.indexOf("i"))) // 和上面相同

    // 2. split, 分解为list (List<String>), 不是Array
    val list = INFO.split(" ")
    println(list)
    val (v1, v2, v3) = list; // 可以使用结构
    println("$v1 - $v2 - $v3") // This - is - a

    // 3. replace
    println(INFO.replace("a", "b"))
    println(INFO.replace(Regex("tes.*"), "bbb")) // 正则表达式
    // 正则表达式, 并且对满足的section进行处理
    println(INFO.replace(Regex("tes.*")) {
        val matchingSection = it.value
        "$matchingSection !!!!"
    })

    // 4. 对比 == 和 ===
    // == 表示内容的比较, 类似于Java的equals, === 表示引用的比较
    val name1 = "Test"
    val name2 = "Test"
    println(name1.equals(name2))
    println(name1 == name2) // true 和上面的相等
    println(name1 === name2) // true, 虽然是使用的是内存地址比较, 但是由于string的特性, 系统会使用string池, 因此实际上内存的位置是相同的.
    val name3 = "test".capitalize()
    // false, 虽然由于内存的string池的特性, 但是这里新得到的string Test其实是根据原始指针的调用. 因此是新开辟的内存空间, 从而和调用者'test'建立关系
    println(name1 === name3)

    val list1 = mutableListOf<String>("test")
    val list2 = mutableListOf<String>("test")
    println(list1 === list2) // false

    // 5. 字符串便利
    INFO.forEach {
        print(" $it ")
    }
    println()

    // 6. 数字安全转换字符
    val number: Int = "666".toInt()
    println(number)
//    val number2: Int = "666.6".toInt();// 错误, 如果无法转换就会异常
    val number2: Int? = "666.6".toIntOrNull() // 对数字进行转换. 否则就会使用null
    println(number2)

    // 6. Double int之间的相互转换
    println(3.14.toInt()) // 3
    println(3.14.roundToInt()) // 3
    println("%.3f".format(3.1415926)) // 保留小数点后面三位
}
```

## 内置函数

### apply 内置函数

```kotlin
/**
 * Apply方法, 对字符串apply一个处理方法
 */
const val INFO = "This is a test String"

fun main() {
    println("length: ${INFO.length}")
    println("last: ${INFO[INFO.length - 1]}")
    println("upper: ${INFO.uppercase(Locale.getDefault())}")

    // apply函数返回的是字符串本身.
    var newInfo: String = INFO.apply {
        // 注意apply的匿名函数参数并不是it, 而是this, 而这个this就是原字符串本身.
        println(this)
        println("length: ${length}")
        println("last: ${this[length - 1]}")
        println("upper: ${uppercase(Locale.getDefault())}")
    }
    println(newInfo)

    // 由于apply返回的是字符串本身, 因此可以循环调用
    INFO.apply {}.apply {}.apply {}
    // 对文件进行配置
    File("study/Note.md").apply {
        setExecutable(true)
        setReadable(true)
    }.apply {
        println(readLines())
    }

}
```

### let内置函数

```kotlin
/**
 * let内置方法, 与apply不同, apply只会返回原始数据, let这会返回处理结果.
 */
fun main() {
    val list: List<Int> = listOf(9, 9, 9, 8, 8, 7, 7) // 创建一个list
    val val1 = list.first() // 获取第一个元素
    val result1 = val1 + val1
    println(result1);

    // 使用let方式
    val let = listOf(9, 9, 9, 8, 8, 7, 7).let {
        println(it) //就是原始list
        it.first() + it.first()// 返回值
    }
    println(let)
}
```

### run内置函数

```kotlin
/**
 * Run 内置函数, 最后一行的类型就是返回值, 相比于let, run使用的结构式this, 而不是it
 */
fun main() {
    val str = "This is a test String"
    var longOrNot = str.run {
        this.length
    }.run(::isLong) // 也可以直接调用function
    println(longOrNot)

    longOrNot = str
        .run { this.length } // 返回长度
        .run { this > 5 } // 长度是否大于5
    println(longOrNot)
}

fun isLong(l: Int) = if (l > 5) true else false
```

### with内置函数

```kotlin
/**
 * With 内置函数, 和run函数类似. 调用方式不同
 */
fun main() {
    val str = "This is a test String"
    val withRes = with(str) {
        // 和run的回调方法一样
        "test"
    }
    println(withRes)

    // 直接打印
    with(str, ::println)
}
```

### also内置函数

```kotlin
/**
 * Also 内置函数, 类似于apply, 返回值不会变.
 */
fun main() {
    val str = "This is a test String"
    var alsoResult = str.also {
        // also使用的是it关键字, 而不是this
        println(it)
    }
    println(alsoResult)
}
```

| 函数名       | 作用                                        | 应用场景                                                                              | 备注                                                                      |
|-----------|-------------------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| let, also | 1. 定义一个变量在特定的域内 <br> 2. 统一做空处理            | 1.明确一个变量所处特定的作用范围内可以使用<br>2. 针对 一个可以null的对象做同意判断空处理                               | 区别在于返回值: <br> * let 函数: 返回值 = 最后一行或者return的表达式 <br> * also返回值 = 传入对象的本身 |
| with      | 调用同一个对象的多个方法/属性时, 可以省去对象名重复, 直接调用方法名/属性即可 | 需要调用一个对象的多个方法/属性                                                                  | 返回值= 函数的最后一行/return表达式                                                  |
| run       | 结合了let函数和with函数的作用                        | 1. 调用同一个对象的多个方法/属性的时候科医生去对象名的重复, 直接调用方法/属性 <br> 2. 定义一个变量在特定作用域内 <br> 3. 统一做判断空处理 | 优点: 避免了let函数必须使用it参数代替对象, 弥补了with函数无法判断空的缺点.                            |                
| apply     | 结合了let函数和with函数的作用                        | 对象实例初始化时需要对对象中的属性进行赋值 & 返回该对象                                                     | 返回传入对象的本身                                                               |

### takeif/takeunless 内置函数

```kotlin
/**
 * takeif 内置方法
 * String可以调用一个回调函数, 如果回调函数结果是false,那么就返回null, 如果是true, 那么就返回当前String
 */
fun main() {
    println("test")
    // 如果回调函数返回是true,那么就返回这个字符串本身
    val userA = "userA".takeIf {
        systemPermission(it, "password")
    }
    println(userA)
    // 如果回调函数的返回值是false那么就返回null
    val userB = "userB".takeIf {
        systemPermission(it, "test")
    }
    println(userB)

    // 一般会配合上空合并操作符
    // 如果存在那么久返回用户名, 如果不存在那么就返回一个错误信息.
    "userC".takeIf {
        systemPermission(it, "userCpwd")
    } ?: "Unauthorized!"
}

fun systemPermission(username: String, password: String): Boolean {
    return username === "root" && password === "password"
}

/**
 * takeUnless 内置方法
 * 和takeif的功能室相仿的, 如果是fasle的就返回name本身, true返回null
 */
fun main() {
    val takeUnless = "user".takeUnless {
//        it.length > 4 // false
        it.isNullOrBlank() // 查看是不是null或者空. 
    }
    println(takeUnless) // 返回user
}
```

## 集合

### list集合

list集合包含两种, 一个是可变集合, 另一个是不可变集合

```kotlin
/**
 * list集合
 */
fun main() {
    val list: List<String> = listOf("a", "b", "test")
    // 索引, 使用的就是运算符重载, 会出现越界错误. 类似于java
    println(list[0])
    // 使用防止下表错误
    println(list.getOrElse(4, { "default" }));
    println(list.getOrNull(4)); //一般配合空合并操作符

    /**
     * 可变/不可变集合
     * listOf 是一个不可变集合
     * mutableListOf 是一个可变集合
     */
    val mutableList = mutableListOf("a")
    mutableList.add("test")
    mutableList.add(1, "b")
    mutableList.add("testa")
    mutableList.remove("testa")
    println(mutableList) // [a, b, test]
    // 将集合变为可变集合
    val list1 = list.toMutableList()
    list1.add("a")
    // 转变为不可变集合
    val immutableList = list1.toList()
    println(immutableList);

    // mutator 函数, 背后就是运算符重载
    mutableList += "newItem1"
    println(mutableList) // [a, b, test, newItem1]
    mutableList -= "newItem1"
    println(mutableList) // [a, b, test]
    // 删除满足条件的元素
    mutableList.removeIf {
        it.contains("a") || it === "A"
    }
    println(mutableList) // [b, test]

    /**
     * List 的遍历
     */
    for (s in list) {
        print("$s ")
    }
    println()
    list.forEach { print("$it ") }
    println()
    list.forEachIndexed { i, item -> print("$item,$i ") }

    /**
     * 解构
     */
    val (value1, value2) = list
    println("$value1,$value2")
    val (_, _, v3) = mutableList // 下划线表示跳过
}
```

### set集合

```kotlin
/**
 * set集合
 * set不会出现重复元素
 */
fun main() {
    val set: Set<String> = setOf("a", "b", "b", "c", "d", "e", "test")
    // set不能使用[]小标的方式提取元素, 但是使用element获取
    // 注意这种方法也是会出现越界的方式
    println(set.elementAt(0))
    val element = set.elementAtOrElse(10) { "default" }
    println(element);
    val eOrNull = set.elementAtOrNull(10)
    println(eOrNull ?: "出现越界")

    /**
     * 可变, set也包括可变和不可变的集合
     */
    val mutableSet = mutableSetOf("a", "b", "b", "c", "d", "e", "test")
    mutableSet += "abc"
    mutableSet -= "test"
    println(mutableSet)

    /**
     * 集合间转换, 比如转换set
     */
    val mutableList = mutableListOf("1", "t", "c", "d", "a", "ahaha", "ahaha")
    val immutableSet = mutableList.toSet() // 变成了一个不可变set
    print(immutableSet)
    // 直接去重
    print(mutableList.distinct())
}
```

### 数组

```kotlin
/**
 * 数组array
 */
fun main() {
    val intArray: IntArray = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9)
    println(intArray.size)
//    println(intArray[10]) // 出现out of index错误
    println(intArray.elementAtOrElse(10) { 9987 })
    println(intArray.elementAtOrNull(10) ?: 9998877)

    // list到array的转换
    val strArray: Array<String> = listOf("a", "b", "c").toTypedArray() // 对象数组
    val intArr: IntArray = listOf(1, 3, 3).toIntArray()

}
```

### map集合

```kotlin
/**
 * Map对象
 */
fun main() {
    // 初始化 可以使用pair
    val map1 = mapOf("a" to 1, "b" to 2, "c" to 3, "d" to 4, "e" to 5)
    val map2 = mapOf(Pair("a", 1), Pair("b", 2), Pair("c", 3), Pair("d", 4), Pair("e", 5))
    println(map1["a"]) // 等同于java的get方法
    println(map1["tt"]) // null

    // 使用默认值的方法
    println(map1.getOrDefault("test", -1)) // 注意第二个参数是一个值而不是函数
    println(map1.getOrElse("test") { -1 })

    // 使用不存在的方法
//    println(map1.getValue("test")) // 不要使用, 会崩溃

    /**
     * map的遍历
     */
    map1.forEach { (k, v) ->
        print("$k,$v ")
    }
    println()
    map1.forEach { entry ->
        print("${entry.key},${entry.value} ")
    }
    println()
    for (entry in map1) {
        print("${entry.key},${entry.value} ")
    }
    println()

    /**
     * 可变map
     */
    val map3 = mutableMapOf("a" to 1, "b" to 2, "c" to 3, "d" to 4, "e" to 5)
    map3.put("f", 6)
    map3.remove("e")
    map3 += "g" to (123)
    map3 += "h" to 321
    map3 -= "h" // 删除h
    map3.forEach { entry ->
        print("${entry.key},${entry.value} ")
    }
    println()

    // 如果ttt存在, 那么就拿出来, 如果没有就放进去然后返回默认值
    val mapRes = map3.getOrPut("ttt") { 123 }
    println(mapRes)
}
```

## 类

函数自带的getter/setter

```kotlin
/**
 * 对gettersetter的覆盖, 使用field关键字
 */
fun main() {
    println(TestClass().info)
}

class TestClass {
    var name = "test"
    var info = "This is a test String"
        get() = field.capitalize()
        set(value) {
            field = "$value!!"
        }
}

/**
 * 如果可能是空, 就必须采用防范静态条件
 */
fun main() {
    println(TestClass().num)
    println(TestClass().num)

    println(TestClass().getShowInfo())
}

class TestClass {
    // 这里不能赋值
    val num: Int // 如果是val的话就会只有get函数不存在set函数
        get() = (1..100).shuffled().first()


    // 防范静态条件
    val info: String? = null
    fun getShowInfo(): String {
        return info?.let { if (it.isBlank()) "Empty" else "Result is $it" } ?: "Empty"
    }
}

/*****************************************************/
/**
 * enum
 */
enum class TestEnum1 {
    A, B, C, D
}

enum class TestEnum2(val num: Int, var info: String) {
    A(1, "a"), B(2, "a"), C(99, "a"), D(321, "a")

    fun get() = num
    fun updateInfo(newInfo: String) {
        this.info = newInfo
    }
}

/*****************************************************/
/**
 * 密封类, 类似于enum但是更强大但是更重
 */
sealed class TestSealed {
    object T1 : TestSealed() // 必须及成本类, 并且使用object
    object T2 : TestSealed()
    object T3 : TestSealed()
    object T4 : TestSealed()
    class T5(val name: String) : TestSealed() // 成员
}

class Tester(val testSealed: TestSealed) {
    fun show() = when (testSealed) {
        is TestSealed.T1 -> "A"
        is TestSealed.T2 -> "B"
        is TestSealed.T3 -> "C"
        is TestSealed.T4 -> "D"
        is TestSealed.T5 -> "E ${testSealed.name}" // 内部类封装
    }
}

fun main() {
    println(Tester(TestSealed.T3).show())
    println(Tester(TestSealed.T5("My Name!")).show())
    // 单例是不能比较的.
    println(TestSealed.T1 === TestSealed.T1)
    // 对象不是唯一的, 因此, 永远是false
    println(TestSealed.T5("test1") === TestSealed.T5("test1"))

}

```

### 构造函数

```kotlin
/**
 * 主构造函数
 */
fun main() {
    println(TestClass(name = "", gender = 'a', age = 123))
    println(TestClass("", 'a', 123))
}

// 那么自动赋值
class TestClass(val name: String /*在输入方直接初始化参数*/, gender: Char, age: Int) {
    var gender = gender
        get() = field
        set(value) {
            field = value
        }
}
/*****************************************************/

/**
 * 次构造函数
 */
// 使用默认赋值, 注意如果所有的次构造函数都是用了默认值赋值, 那么默认使用主构造函数.
class TestClass2(val name: String = "yahaha") {
    // 次构造函数
    // 次构造函数"最终必须"要调用主构造函数 this(name)
    constructor(name: String, age: Int) : this(name) {
    }

    // 在构造函数中复制
    constructor(name: String, age: Int, time: Long = 123L) : this(name, age) {
    }
}

/*****************************************************/
fun main() {
    TestClass3("abc")
}

/**
 * 初始化代码块
 * 一般的次构造函数代码块是不能直接调用未初始化的构造函数的参数的, 但是init代码块可以
 */
class TestClass3(username: String) {
    // 初始化代码块, 注意, 并不是static代码块, 相当于{}构造代码块.
    // 也就是任何构造函数的时候都会执行
    init {
        println("Init block loaded")
        println("User name is : $username")
    }

    constructor(username: String, password: String) : this(username) {}
}
```

构造方法的执行顺序, 当次构造方法被构造:

1. 执行主构造方法, 如果主构造方法包含参数`val/var`那么就初始化出来
2. 初始化类的参数, 也就是root下的`val/var`参数, 同时执行init代码块, 如果代码块内部存在参数, 那么初始化参数
    1. 如果初始化参数写在init代码块后面就执行限制性init代码块
    2. 如果初始化参数写在init代码块前面就先执行初始化参数
3. 执行次构造方法

### 懒加载字段

```kotlin
/*****************************************************/
fun main() {
    val testClass1 = TestClass1()

    testClass1.request()

    testClass1.showData()
}

/**
 * 慢初始化
 */
class TestClass1 {
    lateinit var test: String //必须要使用var, 因为后面进行初始化

    // 类似于懒加载
    fun request() {
        test = "This is a test string"
    }

    fun showData() {
        // println(test)
        // 需要注意的是, 如果没有初始化的话, 哪怕判断语句都会报错.
        // if (test==null) println()

        // 判断是否初始化
        if (::test.isInitialized) {
            println("")
        } else {
            println("Did not init")
        }
    }
}


/*****************************************************/
/**
 * 惰性加载, 自动加载, 在用的时候自动加载
 * 一般属性会在初始化的时候加载, 但是
 */
fun main() {
    val testClass2 = TestClass2()
    println("Before run command!")
    println(testClass2.innerData)
}

class TestClass2 {
    val innerData by lazy { readData() }
    private fun readData(): String {
        println("loading data...")
        println("loading data...")
        println("loading data...")
        return "Data has been loaded!"
    }
}
```

## 类型特性

```kotlin
/*****************************************************/
/**
 * 继承重载关键字的open关键字
 * 所有类都是被final修饰的, 不能被继承, 因此需要关键字open
 */
open class Test1(val name: String) {
    private fun showName() = "Name is : $name"

    // 所有的方法也都是被final修饰的, 因此必须添加open
    open fun myPrintln() = println(showName())
}

class Test2(val subName: String) : Test1(subName) {
    // 复写的话必须添加override关键字
    override fun myPrintln() = println(super.myPrintln())
}

/*****************************************************/
/**
 * 类型转换
 */
open class Test1(val name: String) {
    open fun showName() = "V1 Name is : $name"
}

class Test2(val subName: String) : Test1(subName) {
    override fun showName() = "V2 Name is : $subName"
}

fun main() {
    // 给一个any的定义
    val t: Any = Test2("Test")
    println(t is Test2) // true
    println(t is Test1) // true
    println(t is String) // false

    // 这里调用的是Test2的方法, 因为被重写了
    if (t is Test1) {
        // 这里可以省略"as Test1", is关键字当返回true的时候会自动定义类型
        println((t as Test1).showName())
    }
    if (t is Test2) {
        println((t as Test2).showName())
    }
}
```

> any 超类, 相当于object
> any 类在设计中只提供标准, 不提供实现, 所以看源码没有实现.

```kotlin
/*****************************************************/
/**
 * object修饰的类就是单例
 * 类似于:
 * public static final TestClass INSTANCE;
 */
object TestClass : Any() {
    /**
     * 相比于class, object会使用static
     * static{
     *     TestClass var0 = new TestClass();
     *     INSTANCE = var0;
     *     String var1 = "Init block";
     *     System.out.println(var1);
     * }
     */
    init {
        println("Init block")
    }
}

fun main() {
    println(TestClass.toString())
}

/*****************************************************/
/**
 * 匿名类以及接口
 */
open class TestClass : Any() {
    open fun add(info: String) = println("add $info")
    fun del(info: String) = println("del $info")
}

// kotlin的接口, 注意使用的是functional接口才能使用
fun interface TestInterface {
    fun run(): Unit
}

// 基本接口
interface TestInterface2 {
    fun run(): Unit
}

interface TestInterface3 {
    // 接口可以将get方法覆写
    val version: String
        get() = (1..100).shuffled().last().toString()
    val info: String
        get() = "Static info" // 给一个默认值
}

data class TestClass2(val v: String) : TestInterface3 {
    override val version: String = v // 覆盖了interface的get
    override val info: String
        get() = super.info
}

fun main() {
    // 使用匿名对象的方式进行复写, 也就是object替代了class
    val newTestClass: TestClass = object : TestClass() {
        override fun add(info: String) = println("add2 $info")
    }
    // 使用接口方式
    val r1 = object : Runnable {
        override fun run() {
            println("Test")
        }
    }
    r1.run();

    // 可以直接使用lambda的方式, 类似于上面的编写
    val r2 = Runnable { println("Test") }
    r2.run();

    val r3 = TestInterface { println("Test") }
    r3.run();

    // 如果没有定义functional就必须手动实现方法
    val r4 = object : TestInterface2 {
        override fun run() {
            println("Test")
        }
    }
    r4.run()

    val r5 = TestClass2("321")
    println(r5)
}

/*****************************************************/
/**
 * 抽象类
 */
abstract class TestAbs {
    fun onCreate() {
        setContent(getLayoutId())
    }

    private fun setContent(id: Int) = println("Show ID: ${id}")

    abstract fun getLayoutId(): Int
}


/*****************************************************/
/**
 * 伴生对象
 * 用来定义static静态属性等等. 只会初始化一次, 放入stack
 */
class TestClass {
    companion object {
        // kt会自动生成一个静态类, 然后放出一个getter, 无论是方法还是参数
        val info = "Test String" // public static final String
    }
}

fun main() {
    println(TestClass.info) // 访问静态参数
}

/*****************************************************/
/**
 * 内部类
 *
 */
class TestClass {
    val outerVal = "outer variable";

    // 内部类需要添加inner关键字才能访问外部类
    inner class InnerClass1 {
        fun run() = println("Test: ${outerVal}")
    }

    // 所有的子内部类如果想要访问外部类, 必须所有的都添加Inner标签
    inner class InnerClass2 {
        inner class NestedInnerClass {
            fun run() = println("Test: ${outerVal}")
        }
    }

    // 所有的类默认情况下都是嵌套类
    class Nested {
        fun output() = println("Output String")
    }
}

fun main() {
    // 内部类调用
    TestClass().InnerClass2().NestedInnerClass().run()

    // 嵌套类类似于静态类
    TestClass.Nested().output()
}


/*****************************************************/
/**
 * 数据类, 一般是满足对比类型的方法
 * 一般用于javabean的形式.
 *
 * 可以使用多种特性: 解构, 赋值, hashcode, tostring等方法
 *
 * 注意, copy方法只负责copy字段和重写的方法, 对于次构造方法的复写并不会被copy
 */
data class TestClass(var message: String, var code: Int, var data: String) {

}

fun main() {
    // 可以使用结构
    val (message, code, data) = TestClass("login", 200, "success")
    // TestClass(message=login, code=200, data=success)
    println(TestClass("login", 200, "success"))


    // 所有的数据相同, 返回TRUE
    println(
        TestClass("login", 200, "success")
                == TestClass("login", 200, "success")
    )
    // 不是同一个类, 内存地址不同, ===返回FALSE
    println(
        TestClass("login", 200, "success")
                === TestClass("login", 200, "success")
    )

    val originalClass = TestClass("login", 200, "success")
    // clone字段, 赋值所有字段,同时可以复制新的, 非常推荐使用命名的类型, 如果不使用, 则使用默认类型
    // val copyClass = originalClass.copy(message = "test", code = 123)
    val copyClass = originalClass.copy("test", 123) // TestClass(message=test, code=123, data=success)
    println(copyClass)
}

/*****************************************************/
/**
 * 数据类的自定义解构实现(operator), 可以安排解构赋值的顺序
 * 注意结构组件component[i] 是从1开始的, 并且必须按顺序编写
 */
class TestClass(var message: String, var date: String, var info: String) {
    operator fun component1() = info
    operator fun component2() = message
    operator fun component3() = date
}

fun main() {
    val (info, message, date) = TestClass("a", "b", "c")
}

/*****************************************************/
/**
 * 泛型
 */
class TestGeneric1<T>(private val isR: Boolean, private val obj: T) {
    fun show() = println("Obj: $obj");
    fun getObj() = obj.takeIf { isR }
    fun <B> transfer(item: B) = item

    // 传入lambda方法
    inline fun <R> map(value: T, mapAction: (T) -> R): R = mapAction(value)

    // 类型约束
    fun <R : ParentClass> trans(res: R): R = res
}

open class ParentClass(name: String)
class ChildClass(name: String) : ParentClass(name)

fun main() {
    val testGeneric1 = TestGeneric1(true, "")
    testGeneric1.trans(ChildClass("test"))
}

```

> 数据类的特性:
> 1. 数据类必须至少有一个主构造参数
> 2. 数据类必须有参数, var, val
> 3. 数据类不能使用abstract, open, sealed, inner等等修饰. 数据类就是数据类, 只作为数据处理

```kotlin
/*****************************************************/
/**
 * 动态参数
 * vararg表示接受多个参数.
 */
class TestDynParams(vararg strings: String, val isResult: Boolean) {
    // out表示只读, 不能被修改
    val objarr: Array<out String> = strings

    // 转换
    val objMArr: List<String> = strings.toMutableList()
}

fun main() {
    // 需要添加指定参数isResult=
    TestDynParams("a", "b", isResult = true)
}

/*****************************************************/
/**
 * in/out关键字
 * 协变, 逆变
 */
// 限定泛型T在整个类中只读
interface Producer<out T> {
    fun produce(): T
    // fun consume(item: T) // 错误
}

// 限定只能修改, 不能读取
interface Consumer<in T> {
    fun consume(item: T)
    // fun produce(): T // 错误
}

// 可以任意修改
interface ProducerAndConsumer<T> {
    fun consume(item: T)
    fun produce(): T
}

fun main() {
}
```

> out 父类=子类,
> in 子类=父类

```kotlin
/*****************************************************/
/**
 * reified 关键字
 */
data class TestClass1(val name: String, val age: Int, val study: String)
data class TestClass2(val name: String, val age: Int, val study: String)
data class TestClass3(val name: String, val age: Int, val study: String)

class TestClass {
    // 默认随机产生一个对象, 如果对象和用户对象不一致, 就使用备用对象. 否则返回
    // reified表示让泛型满足判断的能力
    inline fun <reified T> randomOrDefault(getDefault: () -> T): T {
        val objList: List<Any> = listOf(
            TestClass1("a", 1, "Test1"),
            TestClass2("b", 2, "Test2"),
            TestClass3("c", 3, "Test3")
        )
        val rand = objList.shuffled().first()
        return rand.takeIf { it is T } as T?
            ?: getDefault.invoke()
    }
}
```

### 扩展函数

```kotlin
/*****************************************************/
/**
 * 扩展函数, 类的函数扩展
 */
class TestBaseClass(val name: String, val age: Int)

// 对现有类进行扩展, 新建一个函数
fun TestBaseClass.show() {
    // 所有的函数都会有一个this,这个this就是这个对象的本身
    println(this.name)
}

// 即便是java内置类都是可以覆写的
// 如果内置方法被重写就是覆盖
fun String.printThis() {
    println(this)
}

// 对泛型进行扩展, 所有的类型都属于泛型, 因此任何类都可以继承
fun <T> T.show() = println(this)

fun main() {
    TestBaseClass("test", 123).show()
    "test".printThis()
}
/*****************************************************/
/**
 * 对属性进行扩展
 */
val String.myInfo: String
    get() = "My info is here"

/*
   // string属性的扩展使用的参数返回修改
   @NotNull
   public static final String getMyInfo(@NotNull String $this$myInfo) {
      Intrinsics.checkNotNullParameter($this$myInfo, "$this$myInfo");
      return "My info is here";
   }
 */

// 对可空类型的函数扩展
fun String?.getOutput(): String {
    return "Test"
}

fun main() {
    println("Test".myInfo.myInfo.myInfo.myInfo)
    println(null.getOutput())
}
/*****************************************************/
/**
 * infix 关键字, 中缀, 一般配合扩展函数使用, map的to就是扩展函数
 */
infix fun <K, V> K.myTo(value: V): Pair<K, V> {
    return Pair(this, value)
}

fun main() {
    println(mapOf("a" to 1))
    println(mapOf("b".to(2))) // to 方法就是添加了infix关键字
    println(mapOf("c".myTo(3))) // to 方法就是添加了infix关键字
    println(mapOf("d" myTo 4))
}
```

### 扩展方法的文件引入

```kotlin
package com.test.study

fun <T> Iterable<T>.randomValue() = this.shuffled().first()
fun <T> Iterable<T>.randomValue2() = this.shuffled().first()
```

需要注意, 在引入的时候引入的是方法而不是类, 可以使用重命名

```kotlin
import com.test.study.randomValue
import com.test.study.randomValue2 as g

fun main() {
    val list = listOf("a", "b", "c")
    list.randomValue()
    list.g()
}
```

```kotlin
import java.io.File

/*****************************************************/
/**
 * DSL domain specified language 领域专用语言
 *
 * 用来定义输入输出规则.
 * 1. 定义lambda规则标准. 只有特定的类(这个例子中使用的Context)才能使用
 * 2. 同时始终返回调用者本身
 * 3. 匿名函数包含this,同时包含it
 */
class Context {
    val info = "This is a test"
    fun toast(str: String) {
        println("Run the toast: $str")
    }
}

// 在lambda方法中可以直接使用Context中的方法
// 但是注意, 必须public
inline fun Context.myApply(lambda: Context.(String) -> Unit): Context {
    lambda("A new String")
    return this
}

inline fun File.applyFile(processFile: File.(String, String) -> Unit): File {
    processFile(name, absolutePath)
    return this
}

fun main() {
    Context().myApply {
        toast("test") // 使用内部函数
        toast(info) // 使用内部参数
        toast(it) // 使用DSL定义的参数
    }

    File("/tmp").applyFile { name, path ->
        readLines()
        println("$this -> $name");
    }.applyFile { name, path ->
        println("$this -> $path")
    }
}
```

在kotlin中引用java代码:

```java
package com.test.study;

public class Test {
    public static final String data = "This is a test String";
    public static final String nullStr = null;
}
```

```kotlin
import com.test.study.Test

/*****************************************************/
/**
 * 调用java
 */
fun main() {
    val data = Test.data
    println(data.length)
//    val nstr = Test.nullStr
//    println(nstr.length) // 错误, 因为会出现nullexception
    // 正确的方法是在任何时候引用java的时候都要添加null check
    val nstr: String? = Test.nullStr
    println(nstr?.length)
}
```

```kotlin
/*****************************************************/
/**
 * 单例
 *
 * 饿汉式:
 * public final class TestSingleton {
 *    @NotNull
 *    public static final TestSingleton INSTANCE;
 *    private TestSingleton() {
 *    }
 *    static {
 *       TestSingleton var0 = new TestSingleton();
 *       INSTANCE = var0;
 *    }
 * }
 *
 * // 懒汉式
 * public class Test {
 *     private static Test singleton = null;
 *     private Test() {
 *     }
 *     public static Test getTest() {
 *         if (singleton == null) {
 *             singleton = new Test();
 *         }
 *         return singleton;
 *     }
 * }
 *
 */
// 饿汉式
object TestSingleton

// 懒汉式
class TestSingleton2 {
    companion object {
        private var instance: TestSingleton2? = null
            get() {
                if (field == null) {
                    field = TestSingleton2()
                }
                return field;
            }

        // 获取的时候必须要断言, 因为这里返回的可能为null, 但是我们知道构造方法被私有了
        @Synchronized // 同步锁
        fun getInstance() = instance!!
    }
}

// 双重锁
// 直接使用定义将构造方法私有
class TestSingleton3 private constructor() {
    companion object {
        // 使用懒加载同步进行处理
        private val instance: TestSingleton3 by lazy(mode = LazyThreadSafetyMode.SYNCHRONIZED) {
            TestSingleton3()
        }
    }
}

```

## 注解

```kotlin
/*****************************************************/
/**
 * 注解
 */
// jvmName
// 在后端会生成一个类, 也就是这个文件的文件名
// 这行注解需要放在文件的第一行, 也就修改了java的类名, 其他类/方法调用的时候就需要修改名称
@file:JvmName("TestClass")

// jvm field
/*
    在java调用kotlin的时候kotlin的java编译会将下面的代码和field变为私有
    添加了注解以后, jvm会删除"私有", 其他java类就可以读取这个属性
    public final class TestClass1 {
       @JvmField
       @NotNull
       public final String info = "This is a test String";
    }
 */
class TestClass1 {
    @JvmField
    val info = "This is a test String"
}

// jvmOverload
// 默认kotlin会将所有的默认参数私有, 外部java类是无法调用默认参数的
// 添加了以后, 编译器会生成一个重载函数给java类调用用
@JvmOverloads
fun show(a: String = "Test", b: Int = 123) {
}

// JvmStatic
// companion会生成一个静态类. 但是使用java类进行调用就会得不到. 因此需要使用TestClass3.Companion.getTARGET()
// jvm会生成一个getter函数
class TestClass3 {
    companion object {
        @JvmStatic
        val TARGET = "TestCode"

        // 也可以使用field处理, 效果相同
        @JvmField
        val TEST = "TestCode"

        fun show(name: String) = println("$name runs $TARGET!!!!!")
    }
}
```

### Delegate

```kotlin
import kotlin.properties.ReadWriteProperty
import kotlin.reflect.KProperty

/*****************************************************/
/**
 * 代理/委托 delegate
 */
interface DB {
    fun save()
}

class SqlDb() : DB {
    override fun save() {
        println("Save to sql")
    }
}

// 当调用CreateDbAction的方法的时候会使用db, 因为接口被db实现了
class CreateDbAction(db: DB) : DB by db

// 属性委托
class Delegate1() {
    var value1 = 3.14f

    // 属性委托, value2就具备了value1的所有属性/特性, 相当于共用一个getter/setter
    // 本质就是一个单例的调用
    var value2: Float by ::value1
}

// 自定义委托器
class Delegate2() {
    var test: String by CustomDelegate()
}

class CustomDelegate {
    // 实现动态的的委托动作, 主要就是使用的反射, Owner就是反射的体
    operator fun getValue(d2: Delegate2, Property: KProperty<*>): String {
        return "AAA"
    }

    operator fun setValue(d2: Delegate2, Property: KProperty<*>, value: String) {
        println("Set!")
    }
}

// 内置的委托器接口 ReadWriteProperty
class StringDelegate() : ReadWriteProperty<Delegate3, String> {
    override fun getValue(thisRef: Delegate3, property: KProperty<*>): String {
        return "AAA"
    }

    override fun setValue(thisRef: Delegate3, property: KProperty<*>, value: String) {
        println("Set!")
    }
}

// 自定义委托器
class SmartDelegator {
    operator fun provideDelegate(thisRef: Delegate3, property: KProperty<*>): ReadWriteProperty<Delegate3, String> {
        return if (property.name.contains("log")) {
            StringDelegate()
        } else {
            StringDelegate()
        }
    }
}

class Delegate3 {
    var normalText: String by SmartDelegator()
    var logText: String by SmartDelegator()
}

fun main() {
    val delegate2 = Delegate2()
    delegate2.test = "ABCD" // Set!
    println(delegate2.test) // AAA
}
```

