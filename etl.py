import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType as St, StructField as Sf, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Dt, TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config["AWS"]['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config["AWS"]['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = "{}*/*/*/*.json".format(input_data)

    # read song data file
    df = spark.read.json(song_data).dropDuplicates()

    # extract columns to create songs table
    songs_cols = ["title", "artist_id", "year", "duration"]

    # write songs table to parquet files partitioned by year and artist
    songs_table = df.select(songs_cols).dropDuplicates().withColumn("song_id", monotonically_increasing_id())

    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + 'songs/')

    # extract columns to create artists table
    artists_cols = ["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]

    # write artists table to parquet files
    artists_table = df.selectExpr(artists_cols).dropDuplicates()

    artists_table.write.parquet(output_data + 'artists/')

    df.createOrReplaceTempView("df_song_table")


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    users_col = ["userId as user_id", "firstName as first_name", "lastName as last_name", "gender", "level"]

    users_table = df.selectExpr(users_col).dropDuplicates()

    # write users table to parquet files
    users_table = df.selectExpr(users_col).dropDuplicates()

    users_table.write.parquet(output_data + 'users/')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000), TimestampType())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: to_date(x), TimestampType())
    df = df.withColumn("start_time", get_timestamp(col("ts")))

    # extract columns to create time table
    df = df.withColumn("hour", hour("timestamp"))
    df = df.withColumn("day", dayofmonth("timestamp"))
    df = df.withColumn("month", month("timestamp"))
    df = df.withColumn("year", year("timestamp"))
    df = df.withColumn("week", weekofyear("timestamp"))
    df = df.withColumn("weekday", dayofweek("timestamp"))

    time_table = df.select(col("start_time"), col("hour"), col("day"), col("week"), col("month"), col("year"), col("weekday")).distinct()

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy("year", "month").parquet(output_data + 'time/')

    # read in song data to use for songplays table
    song_df = spark.sql("SELECT DISTINCT song_id, artist_id, artist_name FROM df_song_table")

    # extract columns from joined song and log datasets to create songplays table

    songplays_table = df.join(song_df, song_df.artist_name == df.artist, "inner") \
        .distinct() \
        .select(col("start_time"), col("userId"), col("level"), col("sessionId"), \
                col("location"), col("userAgent"), col("song_id"), col("artist_id")) \
        .withColumn("songplay_id", monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy("year", "month").parquet(output_data + 'songplay_table/')


def main():
    spark = create_spark_session()
    input_data_song = "s3a://udacity-dend/song-data/"
    input_data_log = "s3a://udacity-dend/log-data/"

    output_data = "s3://us-west-2.bucket-name/"

    process_song_data(spark, input_data_song, output_data)
    process_log_data(spark, input_data_log, output_data)


if __name__ == "__main__":
    main()
