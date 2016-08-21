#!/usr/bin/python
# -*-encoding:utf-8-*-
'''
@author:www.zencoding.cn
desciption:配置文件
version:v1.0
'''
# 路径配置
fatherPath = r"E:\zen\Documents\data\yhxwfx_data"
originPath = r"E:\zen\Documents\data\yhxwfx_data\userlostprob_train.txt"
trianPath = r"E:\zen\Documents\data\yhxwfx_data\processed_train.csv"
testPath = r"E:\zen\Documents\data\yhxwfx_data\processed_test.csv"
predictPath = r"E:\zen\Documents\data\yhxwfx_data\userlostprob_test.txt"

# 缺失值 按列名处理
# 缺失值超过 一半的 丢弃
dropedfeatures = ['decisionhabit_user',
                  'historyvisit_7ordernum',
                  'historyvisit_totalordernum',
                  'ordercanceledprecent',
                  'firstorder_bu',
                  'historyvisit_visit_detailpagenum'  # 7天内访问酒店详情页数
                  ]
# 确实值 1/5-1/2之间的用-999 代替
fillNauWith999 = ['ordercanncelednum',  # 取消订单数 242114
                  'landhalfhours',  # 24小时登陆时长 28633
                  'starprefer',  # 星级偏好 225053
                  'consuming_capacity',  # 消费能力指数 226108
                  'historyvisit_avghotelnum',  # 近3个月用户历史日均访问酒店数 302069
                  'delta_price1',  # 用户偏好价格-24小时浏览最多酒店价格
                  'businessrate_pre',  # 24小时历史浏览次数最多酒店商务属性指数
                  'ordernum_oneyear',  # 年订单数
                  'avgprice',  # 平均价格
                  'delta_price2',  # 用户偏好价格-24小时浏览酒店平均价格
                  'customer_value_profit',  # 客户近一年的价值
                  'ctrip_profits',  # 客户价值
                  'lasthtlordergap',  # 一年内距离上次下单时长 缺失值占242114条记录
                  'lastpvgap',  # 一年内距上次访问时长 缺失值共97127记录
                  'cr'  # 用户转化率
                  ]

# 缺失值较少的用均值代替
fillNauWithMean = ['commentnums',  # 酒店评论数
                   'novoters',  # 酒店当前评论人数
                   'cancelrate',  # 当前酒店历史取消率 11718
                   'price_sensitive',  # 价格敏感指数
                   'hoteluv',  # 当前酒店历史UV
                   'hotelcr',  # 当前酒店历史转化率
                   'cr_pre',  # 24小时历史浏览次数最多酒店历史cr 29397
                   'lowestprice',  # 当前酒店可定最低价
                   'lowestprice_pre2',  # 24h 访问酒店可预定最低价
                   'customereval_pre2',  # 24小时历史浏览酒店客户评分均值 28633条记录缺失
                   'commentnums_pre',  # 24小时历史浏览次数最多酒店点评数
                   'commentnums_pre2',  # 24小时历史浏览酒店点评数均值
                   'cancelrate_pre',  # 24小时内已访问次数最多酒店历史取消率
                   'novoters_pre2',  # 24小时历史浏览酒店评分人数均值
                   'novoters_pre',  # 24小时历史浏览次数最多酒店评分人数
                   'deltaprice_pre2_t1',  # 24小时内已访问酒店价格与对手价差均值
                   'lowestprice_pre',  # 24小时内已访问次数最多酒店可订最低价
                   'uv_pre',  # 24小时历史浏览次数最多酒店历史uv
                   'uv_pre2',  # 24小时历史浏览酒店历史uv均值
                   'businessrate_pre2',  # 24小时内已访问酒店商务属性指数均值
                   'cityuvs',  # 昨日访问当前城市同入住日期的app uv数
                   'cityorders',  # 昨日提交当前城市同入住日期的app订单数
                   'visitnum_oneyear',  # 年访问次数
                   ]
