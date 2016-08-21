#!/usr/bin/python
#-*-encoding:utf-8-*-
'''
description:数据预处理
@author:www.zencoding.com
version:v1.0
'''
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sb
import matplotlib as plt
from config import dropedfeatures,fillNauWith999,fillNauWithMean,trianPath,fatherPath,originPath,predictPath
import utils

#读取数据
dataOrigin = pd.read_csv(originPath,sep="\t")
predictPath = pd.read_csv(predictPath,sep="\t")
#print(dataOrigin.columns)

# 特征选择
def dropFeature(data,features = dropedfeatures):
    dataDroped = data.drop(features,axis=1) #drop sampleID axis=1 删除列
    return dataDroped

# 缺失值处理
def missValueProcess(data,method="byColumn"):
    '''
    :param method: (1)999 -> 缺失值填充999 ;(2)mean :缺失值填充均值 (3)自定义:确实值没有超过1/2  填充-999 , 缺失值很少则均值替换
    :return: 缺失值处理后的dataFrame
    '''
    if method == "999":
        data.fillna(-999)
    elif method =="mean":
        data.fillna(data.mean())
    elif method == "byColumn":
        for col in fillNauWith999:
            data[col] = data[col].fillna(-999)
        for col in fillNauWithMean:
            fillvalue= (data[col].mean())
            data[col] = data[col].fillna(fillvalue)
    else:
        print("the %s is not supported!"%(method))
    return data

#构造新特征 是否工作日/星期几/预定时间与入住时间间隔
def createNewFeatures(data):
    '''
    @isweekday 1:是 0:否
    @week :星期几 0-6
    @gap : 入住时间-预定时间(天粒度)
    '''
    data['isweekday'] = data['arrival'].map(lambda x:utils.isweekday(x))
    data['week'] = data['arrival'].map(lambda x:utils.datetime2week(x))
    data['gap'] = (pd.to_datetime(data['arrival'])-pd.to_datetime(data['d'])).map(lambda x : (x / np.timedelta64(1, 'D')).astype(int))
    return data

def createUsrTag(dataProcessed):
    '''
    同一个user连续预定了同一家酒店三天，如果第一天的数据（uid+d0）在训练集，第二天和第三天的数据（uid+d1/d2）则会过拟合
    所以我们需要把同一个用户的信息放在同一个数据集中，根据这个思路我们需要自己构建用户标签
    用户标记= hash(用户特征)
    '''
    #user-tag
    dataProcessed['usertag']= dataProcessed.ordercanncelednum\
                                +dataProcessed.historyvisit_avghotelnum \
                                +dataProcessed.ordernum_oneyear \
                                +dataProcessed.customer_value_profit \
                                +dataProcessed.ctrip_profits \
                                +dataProcessed.cr	\
                                +dataProcessed.visitnum_oneyear
    dataProcessed.usertag = dataProcessed.usertag.map(lambda x:hash(x))
    return dataProcessed

def splitTrainAndTest(dataProcessed,percentile =0.75):
    '''
    按用户分组 用一个用户只会出现在训练集 或者 测试集
    :param data:
    :param percent:
    :return:
    '''
    percent =  int(len(dataProcessed.index)*percentile)
    dataProcessed = dataProcessed.sort_values(by="usertag")
    dataProcessed.to_csv(path_or_buf = fatherPath+"\\processed.csv",sep="\t",index=False)
    dataProcessed.iloc[:percent,:].to_csv(path_or_buf = fatherPath+"\\processed_train.csv",sep="\t",index=False)
    dataProcessed.iloc[percent:,:].to_csv(path_or_buf = fatherPath+"\\processed_test.csv",sep="\t",index=False)

if __name__=="__main__":
    dataDroped = dropFeature(data=predictPath)
    createUsrTag(createNewFeatures(missValueProcess(data=dataDroped))).to_csv(path_or_buf= fatherPath+"\\processed_predict.csv",sep="\t",index=False)