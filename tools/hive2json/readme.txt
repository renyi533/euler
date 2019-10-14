This script is used to convert the following hive query result to euler json format.

set mapreduce.reduce.java.opts=-Xmx8092m; 
set mapreduce.reduce.memory.mb=10240;
INSERT OVERWRITE DIRECTORY '/user/tiger/dataplatform/query_editor/yiren/ge_hashid_0822_all'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
select 
    from_user_id, 
    0 as node_type,
    sqrt(count(*)) as out_degree,
    collect_list(to_user_id) as to_id_list,
    collect_list(float(1.0)) as to_weight
from 
(
select from_user_id,to_user_id 
                from relation_table 
                where date=='20190822'
) as follow_table
group by from_user_id
union ALL
select 
    L.to_user_id,
    1 as node_type,
    L.in_degree,
    L.neighbor,
    L.neighbor_weight
from 
    (select to_user_id, sqrt(count(*)) as in_degree,
            collect_set(BIGINT(NULL)) as neighbor,
            collect_set(float(NULL)) as neighbor_weight
                from relation_table 
                where date=='20190822' 
                group by to_user_id) as L
        LEFT OUTER JOIN
    (select DISTINCT from_user_id 
                from relation_table 
                where date=='20190822') as R
        ON L.to_user_id == R.from_user_id 
where R.from_user_id is NULL;   

please set the following environment var
export HADOOP_CLASSPATH=./Hive2Json.jar:$HADOOP_CLASSPATH
export CLASSPATH=./Hive2Json.jar:$CLASSPATH

hadoop jar ./Hive2Json.jar Hive2Json /user/tiger/dataplatform/query_editor/yiren/ge_hashid_0822_all /ss_ml/recommend/yiren/follow/json
