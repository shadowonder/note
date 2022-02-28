# Sqoop

将关系型数据库 oracle , mysql 等数据与 hadoop 数据进行转换的工具

在 apache 下的 sqoop 网站可以下载 sqoop: <https://sqoop.apache.org/>

下载并安装,配置环境变量

配置

1. 在 sqoop 启动的时候会自动加载`/opt/sqoop/conf/sqoop-env.sh`文件. 所以吧 template 文件复制一份放在同样的目录下
2. (可选) 在 `/opt/sqoop/bin/config-sqoop` 文件中,把不必要的 warning 删掉. 134 行左右

   ```shell
   ### Moved to be a runtime check in sqoop.
   #if [ ! -d "${HCAT_HOME}" ]; then
   #  echo "Warning: $HCAT_HOME does not exist! HCatalog jobs will fail."
   #  echo 'Please set $HCAT_HOME to the root of your HCatalog installation.'
   #fi
   #
   #if [ ! -d "${ACCUMULO_HOME}" ]; then
   #  echo "Warning: $ACCUMULO_HOME does not exist! Accumulo imports will fail."
   #  echo 'Please set $ACCUMULO_HOME to the root of your Accumulo installation.'
   #fi
   #if [ ! -d "${ZOOKEEPER_HOME}" ]; then
   #  echo "Warning: $ZOOKEEPER_HOME does not exist! Accumulo imports will fail."
   #  echo 'Please set $ZOOKEEPER_HOME to the root of your Zookeeper installation.'
   #fi
   ```

3. 添加 mysql 的驱动包到 sqoop 的 lib 文件夹下

---

## 操作

使用 sqoop 命令就可以对数据库进行操作

`sqoop list-databases --connect jdbc:mysql://node1:3306/ --username root --password '1qaz!QAZ'` 查看 db 的数据库有哪些

也可以使用脚本文件配置数据库属性, 创建一个脚本文件 getdbs.txt

```text
list-databases
--connect
jdbc:mysql://node1:3306/
--username
root
--password
1qaz!QAZ
```

然后使用命令:`sqoop --options-file getdbs.txt`

---

数据库导入, 从 mysql 中导出一个数据库到 hdfs 中. 最后的-m 指定 mapreduce 的 mapper 的数量

```shell
sqoop import \
--connect jdbc:mysql://node1:3306/mydatabase \
--username root \
--password '1qaz!QAZ' \
--as-textfile \
--table test \
--columns id,name \
--target-dir /sqoop/command \
--delete-target-dir \
-m 1
```

也可以写到文件中:

```text
import
--connect
jdbc:mysql://node1:3306/mydatabase
--username
root
--password
1qaz!QAZ
--as-textfile
--table
test
--columns
id,name
--target-dir
/sqoop/command
--delete-target-dir
-m
1
```

然后启动: `sqoop --options-file sqoop2.txt`

如果出现错误可能是权限问题: `sudo -u hdfs sqoop --options-file sqoop2.txt`

然后就可以查看结果:

```console
[root@node1 sqoopsql]# sudo -u hdfs hdfs dfs -cat /sqoop/command/part-m-00000
1,asdfasef
2,tttttest
3,abasdfasef
4,12312421
```

---

使用 sql 来直接导入数据到 hdfs 中.

> 需要注意的是,在 sql 后面需要加入$CONDITIONS. 否则不能执行

```text
import
--connect
jdbc:mysql://node1:3306/mydatabase
--username
root
--password
1qaz!QAZ
--as-textfile
--target-dir
/sqoop/command3
--delete-target-dir
-m
1
-e
select id,name from test where $CONDITIONS
```

指定分隔符

```text
import
--connect
jdbc:mysql://node1:3306/mydatabase
--username
root
--password
1qaz!QAZ
--as-textfile
--target-dir
/sqoop/dir4
--delete-target-dir
-m
1
-e
select id,name from test where $CONDITIONS
--fields-terminated-by
\t
```

---

### 导入到 hive 中

hive 可以用到的基本配置:
| Argument | Description |
| - | - |
| --hive-home \<dir\> | Override $HIVE_HOME |
| --hive-import | Import tables into Hive (Uses Hive’s default delimiters if none are set.) |
| --hive-overwrite | Overwrite existing data in the Hive table. |
| --create-hive-table | If set, then the job will fail if the target hive |
| | table exits. By default this property is false. |
| --hive-table <table-name> | Sets the table name to use when importing to Hive. |
| --hive-drop-import-delims | Drops \n, \r, and \01 from string fields when importing to Hive. |
| --hive-delims-replacement | Replace \n, \r, and \01 from string fields with user defined string when importing to Hive. |
| --hive-partition-key | Name of a hive field to partition are sharded on |
| --hive-partition-value <v> | String-value that serves as partition key for this imported into hive in this job. |
| --map-column-hive <map> | Override default mapping from SQL type to Hive type for configured columns. |

配置 hive 的 query 文件.这里的 target-dir 配置了 tmp 属性. 主要原因是因为会生成临时文件. 然后基于临时文件进行导入

```text
import
--connect
jdbc:mysql://node1:3306/mydatabase
--username
root
--password
1qaz!QAZ
-m
1
-e
select id,name from test where $CONDITIONS
--hive-import
--create-hive-table
--hive-table
hive_browser_dim
--target-dir
/my/tmp
--delete-target-dir
```

---

### 导出到 mysql 中

这里的 exportdir 指定在 hive 的存储目录中. 同时,mysql 数据库中必须有一个表与其对应. 如果没有需要手动创建这个表

```text
export
--connect
jdbc:mysql://node1:3306/myexporttest
--username
root
--password
1qaz!QAZ
--table
test
--columns
id,name
--export-dir
/user/hive/warehouse/hive_browser_dim
-m
1
--input-fields-terminated-by
\001
```
