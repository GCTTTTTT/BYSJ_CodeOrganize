# -*- coding: utf-8 -*-
import json
from datetime import datetime

from flask import *
from gevent import pywsgi

# from PredictModel import analyse

app = Flask(__name__)


@app.route('/index', methods=['GET'])  # 关于route（）里面可以写url，提交的方式
def indexPage():
    strs = "hello world"
    data = {
        'code': '200',
        'message': 'success',
        'tip': strs
    }
    return jsonify(data)


@app.route('/argmin', methods=['GET'])
def argminIndex():
    strs = "you can analyse you essay here"
    data = {
        'code': '200',
        'message': 'success',
        'tip': strs
    }
    return jsonify(data)


@app.route('/argmin', methods=['POST'])
def argmin():
    print("=======")
    print(request.headers)
    print("=======")

    # data = json.loads(request.form.get('data'))
    # text = data['text']

    # 获取前端json数据
    data = request.get_data()
    print("==========data===========")
    print(data)
    json_data = json.loads(data)
    print("=========json_data===========")
    print(json_data)

    text = json_data.get("txt")
    # password = json_data.get("password")
    # print("text is " + text)
    # print("password is " + password)

    print("==========text===========")
    print(text)
    print("====================")
    # a = request.get_data()  # 获取请求的参数
    # print(a)

    # original_text = json.loads(a)  # 将json数据转换为dict格式
    original_text = text  # 将json数据转换为dict格式
    original_text_len = len(original_text.split())
    print("original_text:")
    print(original_text)
    print("original_text_len:")
    print(original_text_len)
    # 获得前端输入文本传入后端算法函数
    # tag_result = analyse(original_text)
    # 返回数据格式如下，索引为原文本以空格分割文本后的单词索引
    tag_result = {
        'Lead': [
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36']
        ],
        'Position': [
            ['37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
             '53', '54', '55', '56']
        ],
        'Claim': [
            ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70'],
            ['122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134',
             '135', '136', '137', '138', '139', '140']
        ],
        'Evidence': [
            ['71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
             '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
             '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
             '121'],
            ['141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154']
        ],
        'ConcludingStatement': [
            ['155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
             '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181']
        ]
    }
    # Concluding Statement==>ConcludingStatement
    data = {
        'code': '200',
        'message': 'success',
        'original_text': original_text,
        'original_text_len': original_text_len,
        'tagging_text': tag_result,
    }
    print("data:")
    print(data)
    print("jsonify(data)：")
    print(jsonify(data))
    return jsonify(data)


@app.route('/test', methods=['POST'])
def test():
    print("=======")
    print(request.headers)
    print("=======")

    # data = json.loads(request.form.get('data'))
    # text = data['text']

    # 获取前端json数据
    data = request.get_data()
    print("==========data===========")
    print(data)
    json_data = json.loads(data)
    print("=========json_data===========")
    print(json_data)

    text = json_data.get("txt")
    # password = json_data.get("password")
    # print("text is " + text)
    # print("password is " + password)

    print("==========text===========")
    print(text)
    print("====================")
    # a = request.get_data()  # 获取请求的参数
    # print(a)

    # # 给前端传输json数据
    # info = dict()
    # info['status'] = 'success'
    # # info['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # info['text'] = text
    # return jsonify(info)


##########
    # original_text = json.loads(a)  # 将json数据转换为dict格式
    original_text = text  # 将json数据转换为dict格式
    original_text_len = len(original_text.split())
    print("original_text:")
    print(original_text)
    print("original_text_len:")
    print(original_text_len)
    # 获得前端输入文本传入后端算法函数
    # tag_result = analyse(original_text)
    # 返回数据格式如下，索引为原文本以空格分割文本后的单词索引
    tag_result = {
        'Lead': [
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36']
        ],
        'Position': [
            ['37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
             '53', '54', '55', '56']
        ],
        'Claim': [
            ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70'],
            ['122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134',
             '135', '136', '137', '138', '139', '140']
        ],
        'Evidence': [
            ['71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
             '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105',
             '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120',
             '121'],
            ['141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154']
        ],
        'ConcludingStatement': [
            ['155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
             '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181']
        ]
    }
    # Concluding Statement==>ConcludingStatement
    data = {
        'code': '200',
        'message': 'success',
        'original_text': original_text,
        'original_text_len': original_text_len,
        'tagging_text': tag_result,
    }
    print("data:")
    print(data)
    print("jsonify(data)：")
    print(jsonify(data))
    return jsonify(data)

# # 给前端传输json数据
#     info = dict()
#     info['status'] = 'success'
#     # info['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     info['text'] = text
#     return jsonify(info)


# 跨域问题
@app.after_request
def cors(environ):
    environ.headers['Access-Control-Allow-Origin']='*'
    environ.headers['Access-Control-Allow-Method']='*'
    environ.headers['Access-Control-Allow-Headers']='x-requested-with,content-type'
    return environ


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 9999), app)
    server.serve_forever()
