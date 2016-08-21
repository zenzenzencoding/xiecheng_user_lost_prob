# -*- coding: utf-8 -*-
"""
Created on Sat Aug 06 08:30:47 2016

@author: think
"""
import pandas
import numpy as np
def evaluate(evallabel,prob):
    posnum=sum(evallabel)
    
    d=np.c_[evallabel,prob]
    d=pandas.DataFrame(d)
    
    d=d.sort_values(1,ascending=False)
    
    d=d.values
    
    TP=0
    P=0
    
    precision=[]
    recall=[]
    
    for i in range(len(d)):
        if d[i][0]==1:
            TP=TP+1
        P+=1
        precision.append(TP*1.0/P)
        recall.append(TP*1.0/posnum)
    max=0    
    for i in range(len(precision)):
        if precision[i]>0.97:
            if recall[i]>max:
                max=recall[i]
    return max  
  