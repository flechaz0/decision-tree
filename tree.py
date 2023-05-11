from math import log
import operator
import numpy as np

# 数据预处理，删除native-country属性以及缺失信息的数据
def Preprocess(filename):
    data_file = open(filename,'r')
    data_set = []
    for line in data_file.readlines():
        t_array = line.strip('\n').split(', ')
        if '?' in t_array:
            continue
        temp_array1 = np.array(t_array)
        temp_array2 = np.delete(temp_array1, 13)
        t = np.delete(temp_array2, 2)
        data_set.append(t)
    return data_set


def ShannonEnt(data_set):  # 计算给定数据集的香农熵
    numEntries = len(data_set)
    labelCounts = {}
    for featVec in data_set:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(data_set, axis, value):  # 划分数据集,在每次选择完一个特征后删除
    retDataSet = []
    for Vec in data_set:
        featVec = list(Vec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 除去所选的划分属性的一维
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(data_set):  # 选择最好的数据集划分方式
    numFeatures = len(data_set[0]) - 1  # 特征数量
    baseEntropy = ShannonEnt(data_set)  # 原始香农熵
    bestInfoGain = 0.0;
    bestFeature = 0
    for i in range(numFeatures):
        featList = [example[i] for example in data_set]  # 创建一个list存放该特征的所有样本
        uniqueVals = set(featList)  # 从列表中创建一个集合（包含不同的元素）
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(data_set, i, value)
            prob = len(subDataSet) / float(len(data_set))
            newEntropy += prob * ShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if (infoGain >= bestInfoGain):  # 比较，更好则选取
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):  # 多数表决法
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def createTree(dataSet, labels):  # 创建决策树
    labelnum = {'Age':1,'Workclass':2,'Education':3,'EdNum':4,'MaritalStatus':5,
           'Occupation':6,'Relationship':7,'Race':8,'Sex':9,'CapitalGain':10,
           'CapitalLoss':11,'HoursPerWeek':12}
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 所有数据特征相同
    if len(dataSet[0]) == 1:  # 只有一个特征就采用多数表决法
        return majorityCnt(classList)
    bestFeat_index = chooseBestFeatureToSplit(dataSet)  # 选择最好的划分属性的index
    bestFeatLabel = labels[bestFeat_index]              # 最优的label
    myTree = {bestFeatLabel: {}}                        # 用字典形式存储树
    del (labels[bestFeat_index])
    featValues = [example[bestFeat_index] for example in dataSet]
    uniqueVals = set(featValues)
    if len(uniqueVals) == 1:
        return majorityCnt(classList)
    elif len(uniqueVals) < labelnum[bestFeatLabel]:
        myTree[bestFeatLabel]["default"] = majorityCnt(classList)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat_index, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):  # 使用决策树的分类函数
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    if key not in secondDict:
        if "default" in secondDict:
            classLabel = secondDict["default"]
        else:
            classLabel = "<=50K"
        return classLabel
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):  # 保存决策树
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

data_file = 'adult.data'


data_set = Preprocess(data_file)
labels = ['Age','Workclass','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain',
           'CapitalLoss','HoursPerWeek']
tree = createTree(data_set,labels)
storeTree(tree,'abc.pkl')


