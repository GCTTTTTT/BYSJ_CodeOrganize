# dev版 以上实际上在另一个文件
# coding=utf8
import pandas as pd
from flask import *
from flask_cors import CORS  # 导入CORS模块
from gevent import pywsgi
from ProcessCode.survicesImpl import get_data_MIXed_Impl,get_data_ByType_Impl,get_Categorys_Impl,get_randomArt_Impl,get_ArtMain_Impl,get_Search_Impl   # 正式版记得打开这个！！！！
import os 

# 使用Flask创建算法接口
app = Flask(__name__)
# app.json.ensure_ascii = False  # 解决中文乱码问题  flask版本2.3.0以上使用
app.config['JSON_AS_ASCII'] = False  # 解决中文乱码问题  flask版本 2.3.0以下使用

CORS(app)  # 添加跨域支持

# 获取“综合”类别的重要新闻
@app.route('/get_data_mixed', methods=['GET'])
def get_data_MIXed():

    # 获取 URL 参数中的参数值
    date_sel =  request.args.get('dateSel')
    artType = request.args.get('artType')
    page = request.args.get('page')
    pageSize = request.args.get('pageSize')

    result_json = get_data_MIXed_Impl(date_sel)
    print("处理完成！！")

    response = jsonify(result_json)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response  # 将字节串解码为UTF-8字符串


# 按类别获取类别的重要新闻（包括“综合”）
@app.route('/get_data_by_type', methods=['GET'])
def get_data_ByType():
    # 获取 URL 参数中的参数值
    date_sel =  request.args.get('dateSel') # 获取日期参数

    artType = request.args.get('artType')
    page = request.args.get('page')
    pageSize = request.args.get('pageSize')
    
    print(date_sel)
    print(artType)
    print(page)
    print(pageSize)

    # result_json = get_data_ByType_Impl(artType,page,pageSize)
    result_json = get_data_ByType_Impl(date_sel,artType,page,pageSize)

    response = jsonify(result_json)
    print("处理完成！！")

    # response = jsonify(test_json)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    # return response.data.decode('utf-8')  # 将字节串解码为UTF-8字符串
    return response  # 将字节串解码为UTF-8字符串

@app.route('/type', methods=['GET'])
def get_Categorys():
    date_sel =  request.args.get('dateSel') # 获取日期参数
    print(date_sel)
    
    category_list = get_Categorys_Impl(date_sel)
    print("处理完成！！")

    return jsonify(category_list)

# 返回完整文章内容
@app.route('/main', methods=['GET'])
def get_ArtMain():
    
    date_sel =  request.args.get('dateSel') # 获取日期参数

    # 获取artId参数的值
    artId = request.args.get('artId')

    print(date_sel)
    print(artId)
    # result_dict = get_ArtMain(artId)
    result_dict = get_ArtMain_Impl(date_sel,artId) # FIX
    print("处理完成！！")

    response = jsonify(result_dict)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response  # 将字节串解码为UTF-8字符串


# 右侧推荐新闻处 随机返回pageSize条新闻
@app.route('/random_art', methods=['GET'])
def get_randomArt():
    
    date_sel =  request.args.get('dateSel') # 获取日期参数

    # 获取 URL 参数中的参数值
    # artType = request.args.get('artType')
    page = request.args.get('page')
    pageSize = request.args.get('pageSize')
    print(date_sel)

    print(page)
    print(pageSize)

    # result_json = get_randomArt_Impl(page,pageSize)
    result_json = get_randomArt_Impl(date_sel,page,pageSize) # FIX
    
    print("处理完成！！")


    response = jsonify(result_json)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response  # 将字节串解码为UTF-8字符串

# 搜索功能实现
@app.route('/search', methods=['GET'])
def get_data_search():
    # 获取 URL 参数中的参数值
    date_sel =  request.args.get('dateSel') # 获取日期参数

    key = request.args.get('key')
    page = request.args.get('page')
    pageSize = request.args.get('pageSize')
    
    print(date_sel)
    print(key)
    print(page)
    print(pageSize)

    # result_json = get_data_ByType_Impl(artType,page,pageSize)
    result_json = get_Search_Impl(date_sel,key,page,pageSize)

    response = jsonify(result_json)
    print("处理完成！！")

    # response = jsonify(test_json)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    # return response.data.decode('utf-8')  # 将字节串解码为UTF-8字符串
    return response  # 将字节串解码为UTF-8字符串

if __name__ == '__main__':
    # app.run(debug=True)
    server = pywsgi.WSGIServer(('127.0.0.1', 5399), app)
    print("FLASK RUNNING!!")
    server.serve_forever()