#!/usr/bin/python
# -*-encoding:utf-8-*-
'''
@author:www.zencoding.cn
desciption:工具包
version:v1.0
'''
from datetime import datetime
from datetime import time


def datetimeOfdelta(dt0, dt1):
    date0 = datetime.strptime(dt0, "%Y-%m-%d")
    date1 = datetime.strptime(dt1, "%Y-%m-%d")
    return (date1 - date0).days


def datetime2week(dt):
    return datetime.strptime(dt, "%Y-%m-%d").weekday()


def isweekday(dt):
    week = datetime.strptime(dt, "%Y-%m-%d").weekday()
    if int(week) in [0, 1, 2, 3, 4]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    print(datetimeOfdelta("2016-08-06", "2016-08-09"))
    print(datetime2week("2016-08-14"))
    print(isweekday("2016-08-15"))