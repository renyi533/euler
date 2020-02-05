Firstly, you need deploy the three jars to the correct path, for example ~/jar/. 

Then, before run the hadoop job, please correctly set the ENV below.

export JAVA_HOME=/opt/tiger/jdk/jdk1.8/
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$JAVA_HOME/jre/lib/amd64/server/:$LD_LIBRARY_PATH
export HADOOP_HOME=/opt/tiger/yarn_deploy/hadoop
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$PATH
export LIBRARY_PATH=$HADOOP_HOME/lib/native:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH
export CLASSPATH=~/jar/graph_data_parser.jar:~/jar/fastjson-1.2.59.jar:$($HADOOP_HOME/bin/hadoop classpath --glob):$CLASSPATH
export HADOOP_CLASSPATH=${JAVA_HOME}/lib/tools.jar:~/jar/graph_data_parser.jar:~/jar/fastjson-1.2.59.jar:$($HADOOP_HOME/bin/hadoop classpath --glob)

To compile, 

1. mkdir BinaryDataGen; javac -classpath ${HADOOP_CLASSPATH} -d BinaryDataGen/ src/main/BinaryDataGen.java
2. jar -cvf BinaryDataGen.jar -C BinaryDataGen/ .; mv *.jar ./jar/ ; rm -rf BinaryDataGen

Finally, you can run the hadoop job:

hadoop jar <path of BinaryDataGen.jar> BinaryDataGen -libjars <path of fastjson.jar,path of graph_data_parser.jar> <other standard hadoop flags> <input path> <output path> <partition cnt>

for example:

hadoop jar ./BinaryDataGen.jar BinaryDataGen -libjars /data01/home/renyi.bj/jar/fastjson-1.2.59.jar,/data01/home/renyi.bj/jar/graph_data_parser.jar -D mapred.job.priority=HIGH -D mapred.job.map.capacity=100 -D mapred.reduce.tasks=3 -D mapred.job.reduce.capacity=3  -D mapred.job.name='Json to Binary Conversion' -D mapreduce.reduce.java.opts=-Xmx4096M -D mapreduce.map.java.opts=-Xmx4096M -D yarn.cluster.name=xxx -D mapred.job.queue.name=xxx -D mapreduce.map.java.opts=-Xmx24000M  -D mapred.job.map.memory.mb=24000 -D mapred.child.java.opts=-Xmx24000m -D mapreduce.job.reduce.memory.mb=12000 -D mapred.reduce.child.java.opts=-Xmx12000m /ss_ml/recommend/yiren/ppi/input  /ss_ml/recommend/yiren/ppi/out2 4

Please study https://github.com/alibaba/euler/wiki/Preparing-Data for the json format. (Note: for both meta and data, each json block should be in a single line starting with "{" and end with "}".)

All the data json files should be placed under the <input path>. And the meta information should be in  "<input path>/../meta.json". 
