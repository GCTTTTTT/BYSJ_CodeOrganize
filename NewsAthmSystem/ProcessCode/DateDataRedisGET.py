from rediscluster import RedisCluster
import mysql.connector
from datetime import datetime
import json
import pandas as pd
import os

def DateDataGET_Impl(date_sel):
    # 检查文件夹是否存在，如果不存在则创建
    date_sel_folder = f"./datasets/mbert_BiLSTM_SelfAttention/{date_sel}"
    if not os.path.exists(date_sel_folder):
        os.makedirs(date_sel_folder)
    output_file_path = f"{date_sel_folder}/news_{date_sel}.csv"

    # 定义Redis集群的节点信息
    startup_nodes = [
        {"host": "120.25.223.26", "port": "6380"},
        {"host": "120.25.223.26", "port": "6381"},
        {"host": "120.25.223.26", "port": "6382"},
        {"host": "120.25.223.26", "port": "6383"},
        {"host": "120.25.223.26", "port": "6384"},
        {"host": "120.25.223.26", "port": "6385"}
    ]

    # 连接到Redis集群
    rc = RedisCluster(startup_nodes=startup_nodes, decode_responses=True)

    # 连接到MySQL数据库
    # cnx = mysql.connector.connect(user='your_username', password='your_password',
    #                               host='your_host', database='your_database')

    cnx = mysql.connector.connect(
              host="172.16.234.200",
              user="dg_news",
              password="dg_news",
              database="dg_crawler"
            )
    cursor = cnx.cursor()


    # 指定要查询的日期
    # date_sel = "2024-01-01"

    # 检查Redis中是否有指定日期的缓存数据
    cached_data = rc.get(date_sel)

    if cached_data is not None:
        # 如果Redis中有缓存数据,则直接从Redis中获取
        news_data = json.loads(cached_data)
        print(f"Data for {date_sel} retrieved from Redis cache.")
    else:
        # 如果Redis中没有缓存数据,则从MySQL查询数据
        query = f"SELECT * FROM news WHERE pub_time='{date_sel}' AND news.language_id=1779"
        cursor.execute(query)
        news_data = cursor.fetchall()

        # 将查询结果缓存到Redis中
        rc.set(date_sel, json.dumps(news_data, default=str))
        print(f"Data for {date_sel} retrieved from MySQL and cached in Redis.")



    # 将数据转换为DataFrame
    columns = ["id", "website_id", "request_url", "response_url", "category1", "category2",
               "title", "abstract", "body", "pub_time", "cole_time", "images", "language_id", "md5"]
    df = pd.DataFrame(news_data, columns=columns)

    # 保存DataFrame到CSV文件
    # csv_filename = f"news_data_{date_sel}.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Data for {date_sel} saved to {output_file_path}.")

    # 关闭MySQL连接
    cursor.close()
    cnx.close()