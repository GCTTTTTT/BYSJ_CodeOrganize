import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载SBERT模型
model_path = '/root/data/NewsAthm/sentence-transformers/distiluse-base-multilingual-cased-v2'
# model_path = 'distiluse-base-multilingual-cased-v2'
sbert_model = SentenceTransformer(model_path)


# 加载数据
data = pd.read_csv('Data231202-231211.csv')

# 将日期转换为日期时间格式
data['pub_time'] = pd.to_datetime(data['pub_time'])

# 获取唯一日期列表
dates = data['pub_time'].dt.date.unique()

# 设置阈值
threshold = 0.95

# 定义簇列表
clusters = []

# 定义聚类中心更新函数
def update_cluster_center(cluster):
    cluster_embeddings = sbert_model.encode(cluster)
    return np.mean(cluster_embeddings, axis=0)

# 定义写入文件函数
def write_to_file(file_path, clusters):
    with open(file_path, 'w') as file:
        for cluster_info in clusters:
            file.write(f"News Date: {cluster_info['date']}:\n")
            file.write(f"Number of clusters: {len(cluster_info['clusters'])}\n")
            for i, cluster in enumerate(cluster_info['clusters']):
                file.write(f"Cluster {i + 1}:\n")
                file.write(f"Number of news articles: {len(cluster['members'])}\n")
                file.write("News articles:\n")
                for news_article in cluster['members']:
                    file.write(news_article + '\n')
                file.write('\n')

# 对于每个日期
cluster_results = []
for date in dates:
    # 获取该日期的新闻标题
    news_data = data[data['pub_time'].dt.date == date]['title'].tolist()
    
    # 使用SBERT模型获取语义向量
    embeddings = sbert_model.encode(news_data)
    
    # 定义当天的簇列表
    daily_clusters = []
    
    # 对于每个新闻数据
    for i, embedding in enumerate(embeddings):
        # 如果簇列表为空，则新开一个簇
        if not daily_clusters:
            daily_clusters.append({'center': embedding, 'members': [news_data[i]]})
            continue
        
        # 计算当前数据点与各个簇中心的相似度
        similarities = [cosine_similarity([embedding], [cluster['center']])[0][0] for cluster in daily_clusters]
        print(similarities)
        print("==============================================")
        # 找到最大相似度及其对应的簇索引
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        # 如果最大相似度大于阈值，则将当前数据点加入对应簇，并更新簇中心
        if max_similarity > threshold:
            daily_clusters[max_index]['members'].append(news_data[i])
            daily_clusters[max_index]['center'] = update_cluster_center(daily_clusters[max_index]['members'])
        # 否则新开一个簇
        else:
            daily_clusters.append({'center': embedding, 'members': [news_data[i]]})
    
    # 将当天的簇信息添加到结果列表中
    cluster_results.append({'date': date, 'clusters': daily_clusters})

    

file_name = f'single-pass-ByTitle_results_{threshold}.txt'
# 将聚类结果写入到新文件中
write_to_file(file_name, cluster_results)
