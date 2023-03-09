import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

"""References
https://www.cdata.com/kb/tech/mariadb-jdbc-apache-spark.rst
https://sparkbyexamples.com/pyspark/pyspark-join-explained-with-examples/
"""

# Setting up variables for mariadb connection
database = "baseball"
user = "root"
password = input("enter password:")
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class Roll_Avg_Transformer(Transformer):
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, spark, df):
        rolling_avg_dfm = players_rolling_average(spark, df)
        return rolling_avg_dfm


def players_rolling_average(spark, df):
    rolling_avg = spark.sql(
        "SELECT b2.batter, DATE(b2.game_date) AS game_date,"
        " COALESCE(SUM(b1.hit_sum) / NULLIF(SUM(b1.atbat_sum), 0), 0) AS  rolling_avg"
        " FROM batter_data_info AS b2 LEFT JOIN batter_data_info  AS b1"
        " ON ( DATEDIFF(b2.game_date, b1.game_date) <= 100"
        " AND DATEDIFF(b2.game_date, b1.game_date) > 0 AND b2.batter = b1.batter)"
        " GROUP BY b2.batter, b2.game_date ORDER BY b2.batter, b2.game_date"
    )

    rolling_avg.createOrReplaceTempView("rolling_avg")
    rolling_avg.persist(StorageLevel.MEMORY_ONLY)
    return rolling_avg


def read_data(spark, query):
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df.show(3)
    return df


def get_data(spark):
    game_query = "SELECT * FROM game"
    game_data = read_data(spark, game_query)
    game_data.createOrReplaceTempView("game_data")
    game_data.persist(StorageLevel.MEMORY_ONLY)

    batter_count_query = "SELECT * FROM batter_counts"
    batter_count = read_data(spark, batter_count_query)
    batter_count.createOrReplaceTempView("batter_count")
    batter_count.persist(StorageLevel.MEMORY_ONLY)

    batter_data_info = spark.sql(
        "SELECT batter, YEAR(g.local_date) AS game_year, Date(g.local_date) AS game_date, SUM(b.Hit)"
        " AS hit_sum, SUM(b.atBat) AS atBat_sum"
        " FROM batter_count b JOIN game_data g ON b.game_id = g.game_id GROUP BY batter, game_date"
    )

    batter_data_info.createOrReplaceTempView("batter_data_info")
    batter_data_info.persist(StorageLevel.MEMORY_ONLY)

    return batter_data_info


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    batter_data_info = get_data(spark)
    # Calling Rolling Average using transformer
    roll_avg_t: Roll_Avg_Transformer = Roll_Avg_Transformer()
    roll_avg = roll_avg_t._transform(spark, batter_data_info)
    roll_avg.show(10)


if __name__ == "__main__":
    sys.exit(main())
