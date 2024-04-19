# dev实验版 所以堆到一起 以上的代码实际中在另一个文件
import pandas as pd
from ProcessCode.DateDataGET import DateDataGET_Impl              # 正式版记得打开  同级 修改路径
# from ProcessCode.DateDataRedisGET import DateDataGET_Impl              # 正式版记得打开  同级 修改路径

from ProcessCode.DateDataProcess import DateDataProcess_Impl     # 正式版记得打开
# import DateDataGET
# import DateDataProcess
import os

# date_sel要处理成2024-01-01格式
def get_merge_data(date_sel):
    
    model_CLS_name = "mbert_BiLSTM_SelfAttention" ###!!!
    # 读取total_result_pure.csv文件并提取id列的前20条数据以及对应的total_rank字段
    # 总评估处理后文件
    total_result_file_name = f"./MergeFiveDScore/{model_CLS_name}/total_result_pure_{date_sel}_{model_CLS_name}.csv"
    
    # 没有才需要进行处理，加快速度
    if not os.path.exists(total_result_file_name):
        DateDataGET_Impl(date_sel) # 从数据库中获取每日新闻数据 
        DateDataProcess_Impl(date_sel) # 处理每日新闻数据获得分类处理后的_processed和总评估处理后的total_result_pure_
        
    # total_result = pd.read_csv('./datasets/total_result_pure.csv', encoding='utf-8')
    total_result = pd.read_csv(total_result_file_name, encoding='utf-8')

    # id_and_rank = total_result[['id', 'total_rank']].head(20)
    id_and_rank = total_result[['id', 'total_rank']]

    # 分类处理后文件
    processed_data_file_name = f"./datasets/{model_CLS_name}/news_{date_sel}_processed_{model_CLS_name}.csv"

    # news_data = pd.read_csv('./datasets/Data231202_processed.csv', encoding='utf-8')
    news_data = pd.read_csv(processed_data_file_name, encoding='utf-8')

    # 读取website_Rank_new_FIX.csv文件
    website_rank = pd.read_csv('./T2WebsiteRank/website_Rank_new_FIX.csv', encoding='utf-8')

    # 根据id列表在news_data中读取对应id的数据
    merged_data = pd.merge(id_and_rank, news_data, on='id')

    # 根据website_id读取website_Rank_new_FIX.csv文件中的url字段
    merged_data = pd.merge(merged_data, website_rank, on='website_id')


    # 对字符串类型的字段进行编码转换
    def encode_utf8(value):
        if isinstance(value, str):
            return value.encode('utf-8').decode('utf-8')
        return value

    string_columns = ['url', 'request_url', 'response_url', 'category1', 'category2', 'title', 'abstract', 'body', 'images', 'md5']
    merged_data[string_columns] = merged_data[string_columns].applymap(encode_utf8)
    
    return merged_data



def get_data_MIXed_Impl(date_sel):
    
    merged_data = get_merge_data(date_sel)
    # 将每个id对应的数据包装成json格式
    result_json = []
    for index, row in merged_data.iterrows():
        data_dict = {
            'artId': row['id'],
            'total_rank': row['total_rank'],
            'websiteUrl': row['url'],
            'website_id': row['website_id'],
            'request_url': row['request_url'],
            'response_url': row['response_url'],
            'artType': row['category1'],
            # 'category2': row['category2'],
            'artTitle': row['title'],
            'abstract': row['abstract'],
            'artContent': row['body'],
            'artTime': row['pub_time'],
            'cole_time': row['cole_time'],
            'artImageUrl': row['images'],
            'language_id': row['language_id'],
            'md5': row['md5'],
            'artCusId': 582,
            # test
            "customer": {"cusId": 582, "cusName": "admin",
                         "cusPass": None,
                         "cusSpider": "",
                         "cusAvatarUrl": "http://localhost:8080/img/Man.png",
                         "cusStyle": "这个人很懒, 什么都没写",
                         "cusGender": 0,
                         "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
            "artFeature": {"afcId": 710, "afcArtId": 20171864,
                           "afcLikeNum": 0, "afcDislikeNum": 0,
                           "afcComNum": 0, "afcRepNum": 0,
                           "afcReadNum": 0,
                           "afcArtTime": None},
            "cusArtBehavior": None
        }
        result_json.append(data_dict)
        return result_json


def get_data_ByType_Impl(date_sel,artType,page,pageSize):
    
    merged_data = get_merge_data(date_sel)

    
    if artType != "综合":
        # 筛选类别category1为“test”的数据
        filtered_data = merged_data[merged_data['category1'] == artType]
    else:
        filtered_data = merged_data

    # 读取时读全部 然后选择时按照pageSize进行选择显示
    print(type(pageSize))
    # filtered_data = filtered_data.head(int(pageSize))

    # 分页
    # 计算总页数
    total_pages = len(filtered_data) // int(pageSize) + (1 if len(filtered_data) % int(pageSize) > 0 else 0)
    print("total_pages:",total_pages)
    # 计算当前页面的起始索引和结束索引
    start_index = int(page) * int(pageSize)
    end_index = start_index + int(pageSize)
    print("start_index:",start_index)
    print("end_index:",end_index)

    # 获取当前页面的数据
    page_data = filtered_data.iloc[start_index:end_index]
    
    
    # 将每个id对应的数据包装成json格式
    result_json = []
    # for index, row in merged_data.iterrows():
    # for index, row in filtered_data.iterrows():
    for index, row in page_data.iterrows():

        data_dict = {
            'artId': row['id'],
            'total_rank': row['total_rank'],
            'websiteUrl': row['url'],
            'website_id': row['website_id'],
            'request_url': row['request_url'],
            'response_url': row['response_url'],
            'artType': row['category1'],
            # 'category2': row['category2'],
            'artTitle': row['title'],
            'abstract': row['abstract'],
            'artContent': row['body'],
            'artTime': row['pub_time'],
            'cole_time': row['cole_time'],
            'artImageUrl': row['images'],
            'language_id': row['language_id'],
            'md5': row['md5'],
            'artCusId': 582,
            # test
            "customer": {"cusId": 582, "cusName": "admin",
                         "cusPass": None,
                         "cusSpider": "",
                         "cusAvatarUrl": "http://localhost:8080/img/Man.png",
                         "cusStyle": "这个人很懒, 什么都没写",
                         "cusGender": 0,
                         "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
            "artFeature": {"afcId": 710, "afcArtId": 20171864,
                           "afcLikeNum": 0, "afcDislikeNum": 0,
                           "afcComNum": 0, "afcRepNum": 0,
                           "afcReadNum": 0,
                           "afcArtTime": None},
            "cusArtBehavior": None
        }
        result_json.append(data_dict)

    return result_json


def get_Categorys_Impl(date_sel):
    merged_data = get_merge_data(date_sel)

    # # 统计category1不同类别的种类数
    # category_counts = merged_data['category1'].value_counts().to_dict()

    # 提取category1列并转换为列表，并去重
    category_list = merged_data['category1'].unique().tolist()

    # 在列表的最前面加入一个元素"综合"
    category_list.insert(0, '综合')

    return category_list


def get_ArtMain_Impl(date_sel,artId):
    
    merged_data = get_merge_data(date_sel)

    # 根据artId筛选数据
    filtered_data = merged_data[merged_data['id'] == int(artId)]

    # 将每个id对应的数据包装成json格式
    
    result_dict = {}
    # for index, row in merged_data.iterrows():
    for index, row in filtered_data.iterrows():
        data_dict = {
            'artId': row['id'],
            'total_rank': row['total_rank'],
            'websiteUrl': row['url'],
            'website_id': row['website_id'],
            'request_url': row['request_url'],
            'response_url': row['response_url'],
            'artType': row['category1'],
            # 'category2': row['category2'],
            'artTitle': row['title'],
            'abstract': row['abstract'],
            'artContent': row['body'],
            'artTime': row['pub_time'],
            'cole_time': row['cole_time'],
            'artImageUrl': row['images'],
            'language_id': row['language_id'],
            'md5': row['md5'],
            'artCusId': 582,
            # test
            "customer": {"cusId": 582, "cusName": "admin",
                         "cusPass": None,
                         "cusSpider": "",
                         "cusAvatarUrl": "http://localhost:8080/img/Man.png",
                         "cusStyle": "这个人很懒, 什么都没写",
                         "cusGender": 0,
                         "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
            "artFeature": {"afcId": 710, "afcArtId": 20171864,
                           "afcLikeNum": 0, "afcDislikeNum": 0,
                           "afcComNum": 0, "afcRepNum": 0,
                           "afcReadNum": 0,
                           "afcArtTime": None},
            "cusArtBehavior": None
        }

        result_dict = data_dict

    return result_dict

# 右侧推荐新闻处 随机返回pageSize条新闻
def get_randomArt_Impl(date_sel,page,pageSize):
    
    merged_data = get_merge_data(date_sel)

        
    # 随机选取数据
    # print(min(int(pageSize), len(merged_data)))
    sample_data = merged_data.sample(n=min(int(pageSize), len(merged_data)))

    # 将每个id对应的数据包装成json格式
    
    result_json = []
    # for index, row in merged_data.iterrows():
    for index, row in sample_data.iterrows():
        data_dict = {
            'artId': row['id'],
            'total_rank': row['total_rank'],
            'websiteUrl': row['url'],
            'website_id': row['website_id'],
            'request_url': row['request_url'],
            'response_url': row['response_url'],
            'artType': row['category1'],
            # 'category2': row['category2'],
            'artTitle': row['title'],
            'abstract': row['abstract'],
            'artContent': row['body'],
            'artTime': row['pub_time'],
            'cole_time': row['cole_time'],
            'artImageUrl': row['images'],
            'language_id': row['language_id'],
            'md5': row['md5'],
            'artCusId': 582,
            # test
            "customer": {"cusId": 582, "cusName": "admin",
                         "cusPass": None,
                         "cusSpider": "",
                         "cusAvatarUrl": "http://localhost:8080/img/Man.png",
                         "cusStyle": "这个人很懒, 什么都没写",
                         "cusGender": 0,
                         "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
            "artFeature": {"afcId": 710, "afcArtId": 20171864,
                           "afcLikeNum": 0, "afcDislikeNum": 0,
                           "afcComNum": 0, "afcRepNum": 0,
                           "afcReadNum": 0,
                           "afcArtTime": None},
            "cusArtBehavior": None
        }
        result_json.append(data_dict)
        # print(result_json)

    return result_json



# 搜索功能 实现
def get_Search_Impl(date_sel, key,page,pageSize):
    merged_data = get_merge_data(date_sel)
    
    # 筛选出title列或body列中含有关键字的数据
    filtered_data = merged_data[merged_data['title'].str.contains(key) | merged_data['body'].str.contains(key)]
    
    result_json = []
    for index, row in filtered_data.iterrows():
        data_dict = {
            'artId': row['id'],
            'total_rank': row['total_rank'],
            'websiteUrl': row['url'],
            'website_id': row['website_id'],
            'request_url': row['request_url'],
            'response_url': row['response_url'],
            'artType': row['category1'],
            'artTitle': row['title'],
            'abstract': row['abstract'],
            'artContent': row['body'],
            'artTime': row['pub_time'],
            'cole_time': row['cole_time'],
            'artImageUrl': row['images'],
            'language_id': row['language_id'],
            'md5': row['md5'],
            'artCusId': 582,
            "customer": {
                "cusId": 582,
                "cusName": "admin",
                "cusPass": None,
                "cusSpider": "",
                "cusAvatarUrl": "http://localhost:8080/img/Man.png",
                "cusStyle": "这个人很懒, 什么都没写",
                "cusGender": 0,
                "cusTime": "2024-03-14T21:53:09.000+0000",
                "cusLegal": 0
            },
            "artFeature": {
                "afcId": 710,
                "afcArtId": 20171864,
                "afcLikeNum": 0,
                "afcDislikeNum": 0,
                "afcComNum": 0,
                "afcRepNum": 0,
                "afcReadNum": 0,
                "afcArtTime": None
            },
            "cusArtBehavior": None
        }
        result_json.append(data_dict)
    
    return result_json






# import pandas as pd
# from ProcessCode.DateDataGET import DateDataGET_Impl
# from ProcessCode.DateDataProcess import DateDataProcess_Impl

# # date_sel要处理成2024-01-01格式
# def get_merge_data(date_sel):
#     DateDataGET_Impl(date_sel) # 从数据库中获取每日新闻数据 
#     DateDataProcess_Impl(date_sel) # 处理每日新闻数据获得分类处理后的_processed和总评估处理后的total_result_pure_
    
    
#     model_CLS_name = "mbert_BiLSTM_SelfAttention" ###!!!
#     # 读取total_result_pure.csv文件并提取id列的前20条数据以及对应的total_rank字段
#     # 总评估处理后文件
#     total_result_file_name = f"../MergeFiveDScore/{model_CLS_name}/total_result_{date_sel}_{model_CLS_name}.csv"

#     # total_result = pd.read_csv('./datasets/total_result_pure.csv', encoding='utf-8')
#     total_result = pd.read_csv(total_result_file_name, encoding='utf-8')

#     # id_and_rank = total_result[['id', 'total_rank']].head(20)
#     id_and_rank = total_result[['id', 'total_rank']]

#     # 分类处理后文件
#     processed_data_file_name = f"../datasets/{model_CLS_name}/news_{date_sel}_processed_{model_CLS_name}.csv"

#     # news_data = pd.read_csv('./datasets/Data231202_processed.csv', encoding='utf-8')
#     news_data = pd.read_csv(processed_data_file_name, encoding='utf-8')

#     # 读取website_Rank_new_FIX.csv文件
#     website_rank = pd.read_csv('../T2WebsiteRank/website_Rank_new_FIX.csv', encoding='utf-8')

#     # 根据id列表在news_data中读取对应id的数据
#     merged_data = pd.merge(id_and_rank, news_data, on='id')

#     # 根据website_id读取website_Rank_new_FIX.csv文件中的url字段
#     merged_data = pd.merge(merged_data, website_rank, on='website_id')


#     # 对字符串类型的字段进行编码转换
#     def encode_utf8(value):
#         if isinstance(value, str):
#             return value.encode('utf-8').decode('utf-8')
#         return value

#     string_columns = ['url', 'request_url', 'response_url', 'category1', 'category2', 'title', 'abstract', 'body', 'images', 'md5']
#     merged_data[string_columns] = merged_data[string_columns].applymap(encode_utf8)
    
#     return merged_data



# def get_data_MIXed_Impl(date_sel):
    
#     merged_data = get_merge_data(date_sel)
#     # 将每个id对应的数据包装成json格式
#     result_json = []
#     for index, row in merged_data.iterrows():
#         data_dict = {
#             'artId': row['id'],
#             'total_rank': row['total_rank'],
#             'websiteUrl': row['url'],
#             'website_id': row['website_id'],
#             'request_url': row['request_url'],
#             'response_url': row['response_url'],
#             'artType': row['category1'],
#             # 'category2': row['category2'],
#             'artTitle': row['title'],
#             'abstract': row['abstract'],
#             'artContent': row['body'],
#             'artTime': row['pub_time'],
#             'cole_time': row['cole_time'],
#             'artImageUrl': row['images'],
#             'language_id': row['language_id'],
#             'md5': row['md5'],
#             'artCusId': 582,
#             # test
#             "customer": {"cusId": 582, "cusName": "admin",
#                          "cusPass": None,
#                          "cusSpider": "",
#                          "cusAvatarUrl": "http://localhost:8080/img/Man.png",
#                          "cusStyle": "这个人很懒, 什么都没写",
#                          "cusGender": 0,
#                          "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
#             "artFeature": {"afcId": 710, "afcArtId": 20171864,
#                            "afcLikeNum": 0, "afcDislikeNum": 0,
#                            "afcComNum": 0, "afcRepNum": 0,
#                            "afcReadNum": 0,
#                            "afcArtTime": None},
#             "cusArtBehavior": None
#         }
#         result_json.append(data_dict)
#         return result_json


# def get_data_ByType_Impl(date_sel,artType,page,pageSize):
    
#     merged_data = get_merge_data(date_sel)

    
#     if artType != "综合":
#         # 筛选类别category1为“test”的数据
#         filtered_data = merged_data[merged_data['category1'] == artType]
#     else:
#         filtered_data = merged_data

#     # 读取时读全部 然后选择时按照pageSize进行选择显示
#     print(type(pageSize))
#     filtered_data = filtered_data.head(int(pageSize))

#     # 将每个id对应的数据包装成json格式
#     result_json = []
#     # for index, row in merged_data.iterrows():
#     for index, row in filtered_data.iterrows():
#         data_dict = {
#             'artId': row['id'],
#             'total_rank': row['total_rank'],
#             'websiteUrl': row['url'],
#             'website_id': row['website_id'],
#             'request_url': row['request_url'],
#             'response_url': row['response_url'],
#             'artType': row['category1'],
#             # 'category2': row['category2'],
#             'artTitle': row['title'],
#             'abstract': row['abstract'],
#             'artContent': row['body'],
#             'artTime': row['pub_time'],
#             'cole_time': row['cole_time'],
#             'artImageUrl': row['images'],
#             'language_id': row['language_id'],
#             'md5': row['md5'],
#             'artCusId': 582,
#             # test
#             "customer": {"cusId": 582, "cusName": "admin",
#                          "cusPass": None,
#                          "cusSpider": "",
#                          "cusAvatarUrl": "http://localhost:8080/img/Man.png",
#                          "cusStyle": "这个人很懒, 什么都没写",
#                          "cusGender": 0,
#                          "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
#             "artFeature": {"afcId": 710, "afcArtId": 20171864,
#                            "afcLikeNum": 0, "afcDislikeNum": 0,
#                            "afcComNum": 0, "afcRepNum": 0,
#                            "afcReadNum": 0,
#                            "afcArtTime": None},
#             "cusArtBehavior": None
#         }
#         result_json.append(data_dict)

#     return result_json


# def get_Categorys_Impl(date_sel):
#     merged_data = get_merge_data(date_sel)

#     # # 统计category1不同类别的种类数
#     # category_counts = merged_data['category1'].value_counts().to_dict()

#     # 提取category1列并转换为列表，并去重
#     category_list = merged_data['category1'].unique().tolist()

#     # 在列表的最前面加入一个元素"综合"
#     category_list.insert(0, '综合')

#     return category_list


# def get_ArtMain_Impl(date_sel,artId):
    
#     merged_data = get_merge_data(date_sel)

#     # 根据artId筛选数据
#     filtered_data = merged_data[merged_data['id'] == int(artId)]

#     # 将每个id对应的数据包装成json格式
    
#     result_dict = {}
#     # for index, row in merged_data.iterrows():
#     for index, row in filtered_data.iterrows():
#         data_dict = {
#             'artId': row['id'],
#             'total_rank': row['total_rank'],
#             'websiteUrl': row['url'],
#             'website_id': row['website_id'],
#             'request_url': row['request_url'],
#             'response_url': row['response_url'],
#             'artType': row['category1'],
#             # 'category2': row['category2'],
#             'artTitle': row['title'],
#             'abstract': row['abstract'],
#             'artContent': row['body'],
#             'artTime': row['pub_time'],
#             'cole_time': row['cole_time'],
#             'artImageUrl': row['images'],
#             'language_id': row['language_id'],
#             'md5': row['md5'],
#             'artCusId': 582,
#             # test
#             "customer": {"cusId": 582, "cusName": "admin",
#                          "cusPass": None,
#                          "cusSpider": "",
#                          "cusAvatarUrl": "http://localhost:8080/img/Man.png",
#                          "cusStyle": "这个人很懒, 什么都没写",
#                          "cusGender": 0,
#                          "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
#             "artFeature": {"afcId": 710, "afcArtId": 20171864,
#                            "afcLikeNum": 0, "afcDislikeNum": 0,
#                            "afcComNum": 0, "afcRepNum": 0,
#                            "afcReadNum": 0,
#                            "afcArtTime": None},
#             "cusArtBehavior": None
#         }

#         result_dict = data_dict

#     return result_dict

# # 右侧推荐新闻处 随机返回pageSize条新闻
# def get_randomArt_Impl(date_sel,page,pageSize):
    
#     merged_data = get_merge_data(date_sel)

        
#     # 随机选取数据
#     # print(min(int(pageSize), len(merged_data)))
#     sample_data = merged_data.sample(n=min(int(pageSize), len(merged_data)))

#     # 将每个id对应的数据包装成json格式
    
#     result_json = []
#     # for index, row in merged_data.iterrows():
#     for index, row in sample_data.iterrows():
#         data_dict = {
#             'artId': row['id'],
#             'total_rank': row['total_rank'],
#             'websiteUrl': row['url'],
#             'website_id': row['website_id'],
#             'request_url': row['request_url'],
#             'response_url': row['response_url'],
#             'artType': row['category1'],
#             # 'category2': row['category2'],
#             'artTitle': row['title'],
#             'abstract': row['abstract'],
#             'artContent': row['body'],
#             'artTime': row['pub_time'],
#             'cole_time': row['cole_time'],
#             'artImageUrl': row['images'],
#             'language_id': row['language_id'],
#             'md5': row['md5'],
#             'artCusId': 582,
#             # test
#             "customer": {"cusId": 582, "cusName": "admin",
#                          "cusPass": None,
#                          "cusSpider": "",
#                          "cusAvatarUrl": "http://localhost:8080/img/Man.png",
#                          "cusStyle": "这个人很懒, 什么都没写",
#                          "cusGender": 0,
#                          "cusTime": "2024-03-14T21:53:09.000+0000", "cusLegal": 0},
#             "artFeature": {"afcId": 710, "afcArtId": 20171864,
#                            "afcLikeNum": 0, "afcDislikeNum": 0,
#                            "afcComNum": 0, "afcRepNum": 0,
#                            "afcReadNum": 0,
#                            "afcArtTime": None},
#             "cusArtBehavior": None
#         }
#         result_json.append(data_dict)
#         # print(result_json)

#     return result_json