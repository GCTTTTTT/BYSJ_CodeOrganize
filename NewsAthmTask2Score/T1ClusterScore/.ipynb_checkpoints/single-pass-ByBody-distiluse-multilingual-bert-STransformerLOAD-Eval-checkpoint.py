# single-pass-ByBody-distiluse-multilingual-bert-STransformerLOAD-Eval.py
# update:加评估方法，循环阈值进行比较 ，改为index
# update:保存predicted_clusters
# update:3.6 load FIX data
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# yes! 聚类评估！！！可跑 TP, FP, TN, FN 得到RI、Precision、Recall、F1，ARI
# update:单个成簇的处理
from itertools import combinations
from math import comb

def evaluate_clustering(true_clusters, predicted_clusters):
    def count_pairs(cluster):
        pairs = list(combinations(cluster, 2))
        return pairs

    def compute_pairs(true_clusters, predicted_clusters):
        TP, FP, TN, FN = 0, 0, 0, 0
        base_plus = 0
        for true_cluster in true_clusters:
            flag_single = False
            true_cluster = sorted(true_cluster)
            true_pairs = count_pairs(true_cluster) # 排序处理，防止顺序造成偏差
            if len(true_cluster) == 1:
                true_pairs.append((true_cluster[0], true_cluster[0])) # 与自己组成元组
                flag_single = True
            # print("true_cluster:",true_cluster)
            # print("++++++++++++")
            # print("true_pairs:",true_pairs)
            # print("========")
            
            for pair in true_pairs:
                flag = False
                for predicted_cluster in predicted_clusters:
                    predicted_cluster = sorted(predicted_cluster) # 排序处理，防止顺序造成偏差
                    predicted_pairs = count_pairs(predicted_cluster) 
                    if len(predicted_cluster) == 1:
                        predicted_pairs.append((predicted_cluster[0], predicted_cluster[0])) # 与自己组成元组
                    # print("predicted_cluster:",predicted_cluster)
                    # print("++++++++++++")
                    # print("predicted_pairs:",predicted_pairs)
                    if pair in predicted_pairs:
                        TP += 1
                        flag = True # true中同簇，在predict中有找到同簇
                        if flag_single:
                            base_plus+=1
                if not flag: # flag为False，true中同簇，在predict中有找不到同簇
                    FN += 1
                    if flag_single:
                        base_plus+=1
        for predicted_cluster in predicted_clusters:
            flag_single = False
            predicted_cluster = sorted(predicted_cluster)
            predicted_pairs = count_pairs(predicted_cluster) 
            if len(predicted_cluster) == 1:
                predicted_pairs.append((predicted_cluster[0], predicted_cluster[0])) # 与自己组成元组
                flag_single = True
            for pair in predicted_pairs:
                flag2 = False
                for true_cluster in true_clusters:
                    true_cluster = sorted(true_cluster)
                    true_pairs = count_pairs(true_cluster)
                    if len(true_cluster) == 1:
                        true_pairs.append((true_cluster[0], true_cluster[0])) # 与自己组成元组
                    if pair in true_pairs:
                        flag2 = True
                        break
                if not flag2: # flag2为false,在predict中同簇，在true中不同簇(找不到同簇）
                    FP += 1 
                    if flag_single:
                        base_plus+=1
        len_all = 0
        for true_cluster in true_clusters:
            len_all += len(true_cluster)
        print("len_all:",len_all)
        # total_pairs = TP + FP + FN
        # TN = comb(total_pairs, 2) - TP - FP - FN
        TN = comb(len_all, 2) - TP - FP - FN + base_plus # 加base_plus
        # TN = comb(len_all, 2) - TP - FP - FN 
        return TP, FP, TN, FN

    def compute_RI(TP, FP, TN, FN):
        same_cluster_pairs = TP + TN
        different_cluster_pairs = FP + FN

        RI = same_cluster_pairs / (same_cluster_pairs + different_cluster_pairs)
        return RI
    
    def compute_ARI(TP, FP, TN, FN):
        UP = 2 * (TP*TN - FN*FP)
        DOWN = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
        ARI =  UP / DOWN 
        return ARI

    def compute_precision_recall_f(TP, FP, FN):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f_value = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f_value

    # Compute TP, FP, TN, FN
    TP, FP, TN, FN = compute_pairs(true_clusters, predicted_clusters)
    print("TP:",TP)
    print("FP:",FP)
    print("TN:",TN)
    print("FN:",FN)

    # Compute RI
    RI = compute_RI(TP, FP, TN, FN)
    # Compute ARI
    ARI = compute_ARI(TP, FP, TN, FN)

    # Compute precision, recall, F-value
    precision, recall, f_value = compute_precision_recall_f(TP, FP, FN)

    return RI, precision, recall, f_value, ARI

# ==============================================================================

# 加载SBERT模型
model_path = './sentence-transformers/distiluse-base-multilingual-cased-v2'
# model_path = 'distiluse-base-multilingual-cased-v2'
sbert_model = SentenceTransformer(model_path)


# 加载数据
# data = pd.read_csv('Data231202-231211.csv')
data = pd.read_csv('./Data231202-231211_FIX/Data231202_newDATA.csv')


# 将日期转换为日期时间格式
data['pub_time'] = pd.to_datetime(data['pub_time'])

# 获取唯一日期列表
dates = data['pub_time'].dt.date.unique()



# 定义聚类中心更新函数
def update_cluster_center(cluster):
    cluster_embeddings = sbert_model.encode(cluster)
    return np.mean(cluster_embeddings, axis=0)

# 定义写入文件函数
def write_to_file(file_path, clusters,predicted_clusters):
    with open(file_path, 'w') as file:
        file.write(str(predicted_clusters) + '\n')
        file.write("=================================")
        file.write('\n')
        for cluster_info in clusters:
            file.write(f"News Date: {cluster_info['date']}:\n")
            file.write(f"Number of clusters: {len(cluster_info['clusters'])}\n")
            for i, cluster in enumerate(cluster_info['clusters']):
                file.write(f"Cluster {i + 1}:\n")
                file.write(f"Number of news articles: {len(cluster['members'])}\n")
                file.write("News articles:\n")
                # for news_article in cluster['members']:
                #     file.write(news_article + '\n')
                for index in cluster['members']: # 改为index
                    file.write(str(index) + '\n')
                file.write(str(cluster['news']) + '\n')
                file.write("=================================")
                file.write('\n')

# 设置阈值
# threshold = 0.95
# for t in range(90, 100): # 阈值循环
for t in range(1, 10): # 阈值循环
    threshold = 0.99 + (t / 1000.0)
    print(threshold)
    
    # 定义簇列表
    clusters = []
    # 对于每个日期
    cluster_results = []
    cnt = 0
    for date in dates:
        print(cnt)
        cnt+=1
        # # 获取该日期的新闻标题
        # news_data = data[data['pub_time'].dt.date == date]['title'].tolist()
        # 获取该日期的新闻内容body
        news_data = data[data['pub_time'].dt.date == date]['body'].tolist()

        # 使用SBERT模型获取语义向量
        embeddings = sbert_model.encode(news_data)

        # 定义当天的簇列表
        daily_clusters = []

        # 对于每个新闻数据
        for i, embedding in enumerate(embeddings):
            # 如果簇列表为空，则新开一个簇
            if not daily_clusters:
                # daily_clusters.append({'center': embedding, 'members': [news_data[i]]})
                daily_clusters.append({'center': embedding, 'members': [i],'news':[news_data[i]]}) # 改为存index
                continue

            # 计算当前数据点与各个簇中心的相似度
            similarities = [cosine_similarity([embedding], [cluster['center']])[0][0] for cluster in daily_clusters]
            # print(similarities)
            # print("==============================================")
            # 找到最大相似度及其对应的簇索引
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)

            # 如果最大相似度大于阈值，则将当前数据点加入对应簇，并更新簇中心
            if max_similarity > threshold:
                # daily_clusters[max_index]['members'].append(news_data[i])
                daily_clusters[max_index]['members'].append(i) # 改为存index
                daily_clusters[max_index]['news'].append(news_data[i]) # 改为存index
                # daily_clusters[max_index]['center'] = update_cluster_center(daily_clusters[max_index]['members'])
                daily_clusters[max_index]['center'] = update_cluster_center(daily_clusters[max_index]['news']) # 改为传news
            # 否则新开一个簇
            else:
                # daily_clusters.append({'center': embedding, 'members': [news_data[i]]})
                daily_clusters.append({'center': embedding, 'members': [i],'news':[news_data[i]]}) # 改为存index

        # 将当天的簇信息添加到结果列表中
        cluster_results.append({'date': date, 'clusters': daily_clusters})

 #    # 评估
 #    true_clusters = [[0],[1],[2,16],[3],[4,6,22,50,73,87],[5],[7],[8,61],[9],[10,77],[11],[12],[13],
 # [14,29,41,51,59,67,78,84],[15],[17],[18],[19],[20],[21,68],[23],[24],[25],[26],
 # [27],[28],[30],[31],[32],[33],[34],[35,55],[36],[37],[38],[39],[40],[42],[43,64],
 # [44],[45],[46],[47,53,88],[48],[49],[52],[54],[56],[57],[58],[60],[62],[63],[65],
 # [66],[69],[70],[71],[72],[74],[75],[76],[79],[80],[81],[82],[83],[85],[86],
 # [89],[90],[91],[92],[93],[94],[95]]
    
    # update:3.6 新true_clusters for FIX
    true_clusters = [[0],[1,4,6,23,28,41],[2],[3],[5],[7],[8],[9,31],[10],[11],[12],[13],[14],[15],[16],[18],[19],[20],[17,21,38],[22],[24],[25],[26],[27],[29],[30],[32],[33],[34],[35],[36],[37],[39],[40],[42],[43],[44],[45],[46],[47],[48,49],[50],[51],[52],[53],[54],[55],[56],[57],[58],[59],[60],[61],[62],[63],[64],[65],[66],[67],[68,70,74],[69],[71],[72],[73],[75],[76],[77],[78],[79],[80]]
    
    predicted_clusters = []
    for cluster in cluster_results[0]['clusters']: # 2023-12-02的簇s
        clus_index = []
        for i in cluster['members']:
            clus_index.append(i)
        predicted_clusters.append(clus_index)
    print(predicted_clusters)
        

    RI, precision, recall, f_value, ARI = evaluate_clustering(true_clusters, predicted_clusters)
    print("RI:", RI)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-value:", f_value)
    print("ARI:", ARI)
    # 打开文件并追加模式写入
    
    # with open('./results/distiluse-base-multilingual-cased-v2-results/EVAL-single-pass-ByBody-STranformer-LOAD_results.txt', 'a') as file:
    # update: 3.6 FIX
    with open('./results_FIX/distiluse-base-multilingual-cased-v2-results/EVAL-single-pass-ByBody-STranformer-LOAD_results.txt', 'a') as file:
    # with open('./results/angle-bert-base-uncased-nli-en-v1-results/EVAL-single-pass-ByBody_results.txt', 'a') as file: # ByBody
        file.write("threshold: " + str(threshold) + "\n")
        file.write("LOAD BY:" + "STransformer" + "\n")
        file.write("------------------------------------\n")
        file.write("RI: " + str(RI) + "\n")
        file.write("Precision: " + str(precision) + "\n")
        file.write("Recall: " + str(recall) + "\n")
        file.write("F-value: " + str(f_value) + "\n")
        file.write("ARI: " + str(ARI) + "\n")
        file.write("====================================\n")



    # file_name = f'single-pass-ByTitle_results_{threshold}.txt'
    # file_name = f'./results/distiluse-base-multilingual-cased-v2-results/index-single-pass-ByBody-STranformer-LOAD_results_{threshold}.txt'
    # update: 3.6 FIX
    file_name = f'./results_FIX/distiluse-base-multilingual-cased-v2-results/index-single-pass-ByBody-STranformer-LOAD_results_{threshold}.txt'

    # 将聚类结果写入到新文件中
    write_to_file(file_name, cluster_results,predicted_clusters)
