import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    - Returns active Spark sessions for executing application in cluster
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    - Loads songs metadata .json files from input S3 bucket
    - Transforms loaded data into songs and artists dimension tables
    - Writes processed tables into output S3 bucket
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)
    
    # create a temporary view against which you can run SQL queries
    df.createOrReplaceTempView("song_data_table")
    
    # extract columns to create songs table
    songs_table = spark.sql('''
        SELECT DISTINCT song_id, title, artist_id, year, duration
        FROM song_data_table
        WHERE song_id IS NOT NULL
    ''')
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').mode('overwrite').\
                    parquet(output_data + 'analytics/songs/songs.parquet')

    # extract columns to create artists table
    artists_table = spark.sql('''
        SELECT DISTINCT artist_id, artist_name AS name, artist_location as location,
            artist_latitude as latitude, artist_longitude as longitude
        FROM song_data_table
        WHERE artist_id IS NOT NULL;
    ''')
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').\
                    parquet(output_data + 'analytics/artists/artists.parquet')

def process_log_data(spark, input_data, output_data):
    """
    - Loads event log .json files from input S3 bucket
    - Transforms loaded data into dimension (time and users tables) and fact (songplays) tables
    - Writes processed tables into output S3 bucket
    """

    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # create a temporary view against which you can run SQL queries
    df.createOrReplaceTempView("log_data_table")

    # extract columns for users table 
    users_table = spark.sql('''
        SELECT DISTINCT userId as user_id, firstName as first_name, lastName as last_name, gender, level
        FROM log_data_table
        WHERE userId IS NOT NULL
    ''')
    
    # write users table to parquet files
    users_table.write.mode('overwrite').\
                    parquet(output_data + 'analytics/users/users.parquet')
    
    # create and register UDF functions fpr exraxtion datetime columns from original timestamp column
    spark.udf.register("get_st", lambda x: str(datetime.fromtimestamp(x / 1000.0)))
    spark.udf.register("get_year", lambda x: int(datetime.fromtimestamp(x / 1000.0).year))
    spark.udf.register("get_month", lambda x: int(datetime.fromtimestamp(x / 1000.0).month))
    spark.udf.register("get_day", lambda x: int(datetime.fromtimestamp(x / 1000.0).day))
    spark.udf.register("get_hour", lambda x: int(datetime.fromtimestamp(x / 1000.0).hour))
    spark.udf.register("get_week", lambda x: int(datetime.fromtimestamp(x / 1000.0).isocalendar()[1]))
    spark.udf.register("get_weekday", lambda x: int(datetime.fromtimestamp(x / 1000.0).isocalendar()[2]))
    
    # extract columns to create time table
    time_table = spark.sql('''
        SELECT DISTINCT get_st(ts) as start_time, get_year(ts) as year, get_month(ts) as month,
                get_day(ts) as day, get_hour(ts) as hour, get_weekday(ts) as weekday, get_week(ts) as week
        FROM log_data_table
    ''')
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').mode('overwrite').\
                    parquet(output_data + 'analytics/time/time.parquet')
    
    # read in song and data to use for songplays table
    song_df = spark.read.parquet(output_data + 'analytics/songs/songs.parquet')
    artist_df = spark.read.parquet(output_data + 'analytics/artists/artists.parquet')

    # create a temporary view against which you can run SQL queries
    song_df.createOrReplaceTempView("song_data_table")
    artist_df.createOrReplaceTempView("artist_data_table")
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql('''
        SELECT get_st(ld.ts) as start_time, ld.userId as user_id, ld.level as level,
                sd.song_id as song_id, sd.artist_id as artist_id, ld.sessionId as session_id,
                ld.location as location, ld.userAgent as user_agent, get_year(ld.ts) as year, get_month(ld.ts) as month
        FROM log_data_table ld
        JOIN song_data_table sd ON ld.song = sd.title
        JOIN artist_data_table ad ON ad.name = ld.artist
    ''')
    songplays_table = songplays_table.select("*").withColumn("songplay_id", monotonically_increasing_id())
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').mode('overwrite').\
                    parquet(output_data + 'analytics/songplays/songplays.parquet')


def main():
    """
    - Creates Spark sessions

    - Loads original json data from input S3 into Spark cluster.

    - Processes data into fact and dimension tables

    - Writes processed tables into output S3 bucket.
    """
    spark = create_spark_session()
    # State your input s3 bucket with original song and events log json files
    input_data = "s3a://udacity-dend/"
    # State your output s3 bucket for processed fact and dimension tables 
    output_data = "s3a://aws-emr-resources-743815646278-us-east-1/notebooks/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
