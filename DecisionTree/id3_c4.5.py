# -*- coding=utf-8 -*-
#!/usr/bin/python
# ref: http://www.hankcs.com/ml/decision-tree.html

import json
import math
import sys
sys.path.append('.')

def calcShannonEnt(dataSet):
    """
    计算训练数据集中的Y随机变量的香农熵
    @param dataSet:
    @return:
    """
    numEntries = len(dataSet)  # 实例的个数
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
#     print json.dumps(labelCounts, encoding='UTF-8', ensure_ascii=False)
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        # shannonEnt -= prob * log(prob, 2)  # log base 2
        shannonEnt -= prob * math.log(prob, 2)   # log base 2
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    """
    按照给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征的维度
    :param value: 特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 删掉这一维特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    """
    计算X_i给定的条件下，Y的条件熵
    :param dataSet:数据集
    :param i:维度i
    :param featList: 数据集特征列表
    :param uniqueVals: 数据集特征集合
    :return:条件熵
    """
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率(没准也可以用拉普拉斯估计/贝叶斯估计)
        ce += prob * calcShannonEnt(subDataSet)  # ∑pH(Y|X=xi) 条件熵的计算
    return ce


def calcInformationGain(dataSet, baseEntropy, i):
    """
    计算信息增益
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合(去重)
    # 计算每一个取值的条件熵
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain


def calcInformationGainRate(dataSet, baseEntropy, i):
    """
    计算信息增益比(C4.5)
    :param dataSet:数据集
    :param baseEntropy:数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet|X_i)
    """
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy

def chooseBestFeatureToSplitByID3(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGain = calcInformationGain(dataSet, baseEntropy, i)
        if(infoGain > bestInfoGain): # 选择最大的信息增益
            bestInforGain = infoGain
            bestFeature = i
    return bestFeature  # 返回最佳特征对应的维度

def chooseBestFeatureToSplitByC45(dataSet):
    """
    选择最好的数据集划分方式
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # 最后一列是分类
    # print "chooseBestFeatureToSplitByC45", numFeatures, json.dumps(dataSet, encoding='UTF-8', ensure_ascii=False)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有维度特征
        infoGainRate = calcInformationGainRate(dataSet, baseEntropy, i)
        if(infoGainRate > bestInfoGainRate): # 选择最大的信息增益
            bestInfoGainRate = infoGainRate
            bestFeature = i
        # print i, infoGainRate
    # print "bestFeature:", bestFeature
    return bestFeature # 返回最佳特征对应的维度

def createDataSet():
    """
    创建数据集

    :return:
    """
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ]
    labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, labels

def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList: 类列表
    :return: 出现次数最多的类名称
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, chooseBestFeatureToSplitFunc=chooseBestFeatureToSplitByID3):
    """
    创建决策树
    :param dataSet:数据集
    :param labels:数据集每一维的名称
    :return:决策树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 当类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:  # 当只有一个特征的时候，遍历完所有实例返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    print 'bestFeat:', bestFeat, json.dumps(labels, encoding='UTF-8', ensure_ascii=False)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree

if __name__ == "__main__":
    # g_dataset, g_labels = createDataSet()
    # # print g_dataset
    # # print g_labels
    # qingnian_dataset = splitDataSet(g_dataset, 0, u'青年')
    # print u'青年', len(qingnian_dataset)
    # print json.dumps(qingnian_dataset, encoding='UTF-8', ensure_ascii=False)

    ###########中文支持################
    import sys
    # from tree import *

    reload(sys)
    sys.setdefaultencoding('utf-8')
    from pylab import *

    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

    # create test tree
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels)
    # 绘制决策树
    import treePlot

    treePlot.createPlot(myTree)


    # C4.5
    myDat, labels = createDataSet()
    myTree = createTree(myDat, labels, chooseBestFeatureToSplitByC45)
    treePlot.createPlot(myTree)