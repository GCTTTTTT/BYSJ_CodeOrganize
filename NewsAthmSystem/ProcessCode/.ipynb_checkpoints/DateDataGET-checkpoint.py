# 查询数据库中2024-03-02 ~ 2024-03-18的数据用于评估
# todo：包装成一个函数 供外部调用 存储路径和SQL改善
import mysql.connector
import pandas as pd
import os



# date_sel要处理成20xx-01-01格式
def DateDataGET_Impl(date_sel):
    # 检查文件夹是否存在，如果不存在则创建
    date_sel_folder = f"./datasets/mbert_BiLSTM_SelfAttention/{date_sel}"
    if not os.path.exists(date_sel_folder):
        os.makedirs(date_sel_folder)

    
    output_file_path = f"{date_sel_folder}/news_{date_sel}.csv"
    # 缓存处理 不存在才需要重新查询保存
    if not os.path.exists(output_file_path):
        # df.to_csv(output_file_path, index=False)
        # 连接数据库
        conn = mysql.connector.connect(
          host="172.16.234.200",
          user="dg_news",
          password="dg_news",
          database="dg_crawler"
        )


        # 执行SQL查询
        query = """
            SELECT *
            FROM news
            WHERE pub_time='"""+date_sel+"""' and news.language_id=1779
        """

        # 将查询结果存入DataFrame
        df = pd.read_sql(query, conn)

        # 关闭数据库连接
        conn.close()

        # 将DataFrame写入CSV文件
        # 按天文件夹

#         # 检查文件夹是否存在，如果不存在则创建
#         date_sel_folder = f"./datasets/mbert_BiLSTM_SelfAttention/{date_sel}"
#         if not os.path.exists(date_sel_folder):
#             os.makedirs(date_sel_folder)

#         output_file_path = f"{date_sel_folder}/news_{date_sel}.csv"
        df.to_csv(output_file_path, index=False)
              
    
              
              
