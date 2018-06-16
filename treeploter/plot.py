
# coding: utf-8

# In[4]:
import numpy as np
import operator

from math import log
def calcShannonEnt(dataSet):#计算香农熵
    numEntries=len(dataSet)#获取数据集的文件数目
    labelCounts={}#建立一个标签矩阵，以存放对应的标签值
    #依次在数据集当中將数据集的标签放到标签矩阵里
    for featVec in dataSet:
        currentLabels=featVec[-1]#依次取featVec里的最后一个元素，即标签元素
        if currentLabels not in labelCounts.keys():#如果得到的标签不在已有的标签集内
            labelCounts[currentLabels]=0#创建一个新的标签
        labelCounts[currentLabels]+=1#对对应的标签种类进行计数
    shannonEnt=0.0#熵清0
    #对所有不同类别计算熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt


# In[9]:


def createDataSet():
    dataSet=([[1,1,'Yes'],[1,1,'Yes'],[1,0,'No'],[0,1,'No'],[0,1,'No']])#这里不能写array，因为不是纯数
    labels=['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):#按照给定特征对数据集进行划分，三个参量依次是数据集，划分数据集的特征和特征返回值
    retDataSet=[]#构建新矩阵
    #对于数据集中的数据
    for featVec in dataSet:
        if featVec[axis]==value:#一旦发现特征与目标值一致
            reducedFeatVec=featVec[:axis]#將axis之前的列附到resucdFeatVec里
            reducedFeatVec.extend(featVec[axis+1:])#将axis以后的列附到reducedFeatVec里
            retDataSet.append(reducedFeatVec)#將去除了指定特征列的数据集放在retDataSet里
    return retDataSet#返回划分后数据集

def chooseBestFeatureToSplit(dataSet):#选择最好的数据分割点
    lens=len(dataSet[0])-1#对数据集第一行，即第一个例子的行数（特征个数），减1是为了方便数组计算
    baseShannonEnt=calcShannonEnt(dataSet)#算出原始香农熵
    bestGain=0.0#將最信息增益先置零
    bestFeature=-1#將最佳分割点值-1，不置0是因为0相当于第一个分割点，会引起误解
    for i in range(lens):#循环各个特征找使信息熵最大点处
        featList=[example[i] for example in dataSet]#对于数据集里的所有特征遍历
        uniqueVals=set(featList)#设置一个列表存放这些特征
        newEnt=0.0#信息熵清零
        for value in uniqueVals:#遍历所有特征，根据特征不同来实现分割从而获取不同信息熵
            subDataSet=splitDataSet(dataSet,i,value)#不同特征处分割
            prob=len(subDataSet)/float(len(dataSet))
            newEnt+=prob*calcShannonEnt(subDataSet)#获取此时的信息熵
        infoGain=baseShannonEnt-newEnt#获取信息增益
        if(infoGain>bestGain):#如果信息增益比现有最好增益还大
            bestGain=infoGain#则取代他
            bestFeature=i#并记下此时的分割位置
    return bestFeature#返回分割位置

#####################下述两个函数组合在一起就是ID3决策树算法#############################
#####################这个多数表决程序本质上与K近邻法的第一个分类器是一样##################
            
def majorityCnt(classList):#多数表决分类函数
    classCount={}#建立一个数据字典，里面存储所有的类别
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0#如果有新的类别，则创立一个新的元素代表该种类
        classCount[vote]+=1#否则该元素加一
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#对数据集进行排序，第二行作为排序依据，从高到低排   
    return sortedClassCount[0][0]#把第一个元素返回，即返回出现次数最多的那个元素

def createTree(dataSet,labels):#创建一个决策树
    classList=[example[-1] for example in dataSet]#以数据集的最后一列作为新的一个列表
    if classList.count(classList[0])==len(classList):#如果分类列表完全相同
        return classList[0]#停止继续划分
    if len(dataSet[0])==1:#如果遍历完所有特征，仍不能划分为唯一门类
        return majorithCnt(classList)#返回出现出现次数最多的那个类标签
    bestFeat=chooseBestFeatureToSplit(dataSet)#否则选择最优特征
    bestFeatLabel=labels[bestFeat]#同时將最优特征的标签赋予bestFeatureLabel
    myTree = {bestFeatLabel:{}}#根据最优标签生成树
    del(labels[bestFeat])#將刚刚生成树所使用的标签去掉
    featValues=[example[bestFeat] for example in dataSet]#获取所有训练集中最优特征属性值
    uniqueVals=set(featValues)#把重复的属性去掉，并放到uniqueVals里
    for value in uniqueVals:#遍历特征遍历的所有属性值
        subLabels=labels[:]#先把原始标签数据完全复制，防止对原列表干扰
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)#再以该特征来划分决策树
    return myTree#返回决策树
