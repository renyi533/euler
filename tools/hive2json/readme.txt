This script is used to convert the following hive query result to euler json format.

tab separated format for:
node id, type, weight, <target node list 1>, <target node weight list 1>, ....., <target node list N>, <target node weight list N>

please set the following environment var
export HADOOP_CLASSPATH=./Hive2Json.jar:$HADOOP_CLASSPATH
export CLASSPATH=./Hive2Json.jar:$CLASSPATH

jar ./Hive2Json.jar Hive2Json -D mapreduce.job.reduce.memory.mb=12000 -D mapred.job.map.memory.mb=12000 -D mapred.child.java.opts=-Xmx6000m -D mapred.reduce.child.java.opts=-Xmx6000m -D yarn.cluster.name=xxx -D mapred.job.queue.name=xxxx <input path>  <output path>
