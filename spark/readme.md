sc = pyspark.SparkContext('yarn')

a = sc.textFile("hdfs://master:9000/Data/Abcnews-date-text.csv")


Cd $HADDOP_HOME
Cd /usr/loal/hadoop/etc
Cat core-site.xml

hdfs://master:9000

hdfs://master:9000<
/Data/Binance_Parquet



Here is the solution
sc.textFile("hdfs://nn1home:8020/input/war-and-peace.txt")
How did I find out nn1home:8020?
Just search for the file core-site.xml and look for xml element fs.defaultFS

From <https://stackoverflow.com/questions/27478096/cannot-read-a-file-from-hdfs-using-spark> 

or me config file was at $HADOOP_HOME/etc/hadoop/core-site.xml

From <https://stackoverflow.com/questions/27478096/cannot-read-a-file-from-hdfs-using-spark