#!/usr/bin/python
#-*-encoding:utf-8-*-
'''
description:gbdt模型
@author:www.zencoding.cn
version:v1.0
'''
from __future__ import division
import numpy as np
import pandas as pd
import time
import sys
from config import fatherPath,trianPath,testPath
import logging
from sklearn import metrics
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=fatherPath+'/test.log',
                    filemode='w')
logger = logging.getLogger(__name__)

def trainMethod(trian=trianPath, test=testPath,method="gbdt"):
    '''
    :param trian: 训练数据集路径
    :param test: 测试数据集路径
    :param method: gbdt\gbdt-lr
    :return: None
    '''
    logger.info('[INFO]:GBDT算法-正在加载数据...')
    trianData = pd.read_csv(trian,sep="\t").drop(["usertag","d","arrival"],axis=1)
    testData = pd.read_csv(test,sep="\t").drop(["usertag","d","arrival"],axis=1)
    trian_X = trianData.iloc[:,1:]
    train_Y = trianData.iloc[:,0]
    test_X = testData.iloc[:,1:]
    test_Y = testData.iloc[:,0]
    dtrain = xgb.DMatrix(trian_X.values,label=train_Y.values)
    dtest = xgb.DMatrix(test_X.values,label=test_Y.values)
    logger.info('[INFO]:GBDT算法-完成加载数据！')
    # training parameter
    param = {'bst:max_depth': 5, 'bst:eta': 0.05, 'silent': 0, 'objective': 'binary:logistic'}
    param['nthread'] = 8
    param['eval_metric'] = ['logloss', 'auc']
    param['subsample'] = 0.5
    param['num_feature'] = 'sqrt'
    num_round = 1000

    eval_list = [(dtest, 'dtest'), (dtrain, 'train')]

    logger.info('[INFO]:GBDT算法-正在模型训练...')
    bst = xgb.train(param, dtrain, num_round, eval_list, early_stopping_rounds=100)
    logger.info('[INFO]:GBDT算法-模型训练完成！')

    # dump model file and feature importance
    # bst.save_model(output_path + '/xgboost_clf_' + version + '.model')
    logger.info('[INFO]:feature_importances')
    feature_importances = pd.Series(bst.get_fscore())
    print(feature_importances.sort_values())
    #feature_importances.sort_values(by=, ascending=False, inplace=True)
    feature_importances.sort_values().to_csv( fatherPath + '/feature_importances.csv')

    #print ('best_score is %5.6f' % bst.best_score)
    #print ('best_iteration is %5d' % bst.best_iteration)
    #print ('best_ntree_limit is %5d' % bst.best_ntree_limit)

    bst.set_param({'nthread': 1})
    clf_threshold = 0.5
    if method == "gbdt":
        logger.info('[INFO]:GBDT算法-模型正在预测...')
        test_predict = bst.predict(dtest)
        test_predict_label = np.fromiter(map(lambda x: 1 if x > clf_threshold else 0, test_predict),dtype=np.int)
        test_label = dtest.get_label()
        auc = metrics.roc_auc_score(test_label, test_predict, sample_weight=None)
        print(test_label)
        print(test_predict)
        print(test_predict_label)
        precision,recall,thresholds = precision_recall_curve(test_label,test_predict)
        print(precision,recall,thresholds)
        pr = pd.DataFrame({"precision":precision,"recall":recall})
        prc= pr[pr.precision>=0.97].recall.max()
        print("AUC is %5.6f" % (auc))
        print("classification report")
        print(metrics.classification_report(list(test_label), test_predict_label))
        print('confusion matrix')
        print(metrics.confusion_matrix(test_label, test_predict_label))
        print('accuracy')
        print(metrics.accuracy_score(test_label, test_predict_label))
        print('roc')
        print(metrics.roc_curve(test_label, test_predict_label))
        print('precision-recall-curve')
        print(prc)
        logger.info("[INFO] precision-recall-curve:%s"%(prc))


    if method=="gbdt-lr":
        logger.info("[INFO] GBDT-LR 算法:正在训练...")
        ndata = bst.predict(dtrain,pred_leaf=True)
        enc = OneHotEncoder()
        enc.fit(ndata)
        ntrain = enc.transform(ndata).toarray()
        lr = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.01, fit_intercept=True,
                intercept_scaling=1, solver='liblinear', max_iter=100)
        lr.fit(ntrain, dtrain.get_label())
        ntest = enc.transform(bst.predict(dtest,pred_leaf=True))

        logger.info("[INFO] GBDT-LR 算法:正在预测...")
        test_predict = lr.predict_proba(ntest)
        test_predict_label = lr.predict(ntest)
        auc = metrics.roc_auc_score(test_label, test_predict[:,1], sample_weight=None)
        precision, recall, _ = precision_recall_curve(test_label, test_predict_label)
        pr = pd.DataFrame({"precision": precision, "recall": recall})
        prc = pr[pr.precision >= 0.97].recall.max()
        print ("lr AUC is %5.6f" % (auc))
        print ('lr classification report')
        print (metrics.classification_report(list(test_label), test_predict_label))
        print (metrics.classification_report(list(test_label), test_predict_label))
        print ('lr confusion matrix')
        print (metrics.confusion_matrix(test_label, test_predict_label))
        print ('lr accuracy')
        print (metrics.accuracy_score(test_label, test_predict_label))
        print ('lr roc')
        print (metrics.roc_curve(test_label, test_predict_label))
        print('precision-recall-curve')
        print(prc)
        logger.info("[INFO] precision-recall-curve:%s" % (prc))
        logger.info("[INFO] GBDT-LR 算法:结束...")

if __name__ == '__main__':
    trainMethod()