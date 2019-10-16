import java.io.IOException;
import java.util.*;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;
import java.lang.Math;

public class Hive2Json { 	
    public static class JsonMap extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
      private Text value_word = new Text();
      private Text key_word = new Text();
      private static String pass2line(List<String> fields){
        String from_id = fields.get(0);
        String type = fields.get(1);
        String weight = fields.get(2);
        int edge_type_cnt = (fields.size() - 3)/2;
        int offset = 3;

        ArrayList<ArrayList<String>> nb_lists = new ArrayList<ArrayList<String>>();
        ArrayList<ArrayList<String>> nb_weight_lists = new ArrayList<ArrayList<String>>();
        StringBuilder line = new StringBuilder(
                "{\"node_id\": " + from_id + ", " +
                "\"node_type\": " + type + ", " +
                "\"node_weight\": " + weight + ", " +
                "\"neighbor\": { " );
        for (int i=0; i<edge_type_cnt; i++) {
          line.append("\"" + Integer.toString(i) + "\": {"); 
          String neighbor = fields.get(offset + 2*i);
          String neighbor_weight = fields.get(offset + 2*i + 1);
          StringTokenizer neighbor_tokenizer = 
              new StringTokenizer(neighbor, " ,[]\t\"\1\2");
          StringTokenizer neighbor_weight_tokenizer = 
              new StringTokenizer(neighbor_weight, " ,[]\t\"\1\2");

          ArrayList<String> nb_list = new  ArrayList<String>();
          ArrayList<String> nb_weight_list = new  ArrayList<String>();

          while (neighbor_tokenizer.hasMoreTokens() && 
                  neighbor_weight_tokenizer.hasMoreTokens()) {
              nb_list.add(neighbor_tokenizer.nextToken());
              nb_weight_list.add(neighbor_weight_tokenizer.nextToken());
          } 

          nb_lists.add(nb_list);
          nb_weight_lists.add(nb_weight_list);

          for(int j=0; j<nb_list.size(); j++) { 
              String to_id = nb_list.get(j);
              String nb_weight = nb_weight_list.get(j);   
              line.append( "\"" + to_id + "\":" + nb_weight); 
              if (j != nb_list.size()-1) 
                  line.append( ", ");
          }
          line.append("}");
          if (i < edge_type_cnt-1) {
            line.append(", ");
          } 
        }
        line.append("}, ");
        line.append("\"uint64_feature\":{}, \"float_feature\":{}, \"binary_feature\":{}, "
                    + "\"edge\":[");
        
        for (int j=0; j<edge_type_cnt; j++) {         
          List<String> nb_list = nb_lists.get(j);
          List<String> nb_weight_list = nb_weight_lists.get(j);
          for(int i=0; i<nb_list.size(); i++) { 
              String to_id = nb_list.get(i);
              String nb_weight = nb_weight_list.get(i);   
              line.append("{\"src_id\":" + from_id 
                          + ", \"dst_id\":" + to_id 
                          + ", \"edge_type\":"+ Integer.toString(j) +", \"weight\":"+nb_weight+", \"uint64_feature\":{}, "
                          + "\"float_feature\":{}, \"binary_feature\":{}}"); 
              if (j != edge_type_cnt-1 || i != nb_list.size()-1) 
                  line.append(", ");
          }
        }
        line.append( "]}");
        return line.toString();
      }

      public void map(LongWritable key, 
                      Text value, 
                      OutputCollector<Text, Text> output, 
                      Reporter reporter) throws IOException {
        List<String> words = new ArrayList<>();
        String line = value.toString();
        String[] items= line.split("\t", -1);
        int i = 0;
        while (i < items.length) {
           words.add(items[i++]);
        }
        String from_id = words.get(0);
        if (!from_id.equals("from_id")) {
          key_word.set((pass2line(words)));
          value_word.set("");
          output.collect(key_word, value_word);
        }
      }
   }

   public static class JsonReduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {
        
       public void reduce(Text key, 
                        Iterator<Text> value, 
                        OutputCollector<Text, Text> output, 
                        Reporter reporter) throws IOException {
        output.collect(key, value.next());
     }
   }

   public static void main(String[] args) throws Exception {
     JobConf conf = new JobConf(Hive2Json.class);
     conf.setJobName("hive2json");
     
     conf.setJarByClass(Hive2Json.class);

     conf.setOutputKeyClass(Text.class);
     conf.setOutputValueClass(Text.class);
     
     conf.setMapperClass(JsonMap.class);
     conf.setCombinerClass(JsonReduce.class);
     conf.setReducerClass(JsonReduce.class);

     conf.setInputFormat(TextInputFormat.class);
     conf.setOutputFormat(TextOutputFormat.class);
	 conf.setNumReduceTasks(1000);

     FileInputFormat.setInputPaths(conf, new Path(args[0]));
     FileOutputFormat.setOutputPath(conf, new Path(args[1]));

     JobClient.runJob(conf);
   }
}
 
