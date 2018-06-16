from numpy import *#载入numpy库
import numpy as np
import operator#载入operator模块
from os import listdir#从os模块导入listdir，可以给出给定目录文件名
from sklearn.neighbors import KNeighborsClassifier as kNN#载入sklearn库



def createDataSet():#建立数据集
  group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
  labels=['A','A','B','B']
  return group,labels

def classify0(inX,dataSet,labels,k):#该函数为简单kNN分类器
  #首先计算已知类别数据集与当前点的距离
  dataSetSize=dataSet.shape[0] #读取数据集的行数，并把行数放到dataSetSize里，shape[]用来读取矩阵的行列数，shape[1]表示读取列数
  diffMat=tile(inX,(dataSetSize,1))-dataSet #tile(inX,(dataSetSize,1))复制比较向量inX，tile的功能是告诉inX需要复制多少遍，这
  #里复制成(dataSetSize行，一列)目的是把inX转化成与数据集相同大小，再与数据集矩阵相减，形成的差值矩阵存放在diffMat里
  sqDiffMat=diffMat**2#注意这里是把矩阵李的各个元素依次平方，如（[-1,-1.1],[-1,-1]）执行该操作后为（[1,1.21],[1,1]）
  sqDistances=sqDiffMat.sum(axis=1)#实现计算计算结果，axis表矩阵每一行元素相加，如（[1,1.21],[1,1]）,执行该操作后为（2.21，2）
  distances=sqDistances**0.5#开根号
  #按照距离递增次序排序
  sortedDisIndicies=distances.argsort()#使用argsort排序，返回从小到大到“顺序值”
  #如{2,4,1}返回{1,2,0}，依次为其顺序到索引
  classCount={}#新建一个字典，用于计数
  #选取与当前点距离最小的k个点
  for i in range(k):#按顺序对标签进行计数
   voteIlabel=labels[sortedDisIndicies[i]]#按照之前排序值依次对标签进行计数
   classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#对字典进行抓取，此时字典是空的
  #所以没有标签，现在將一个标签作为key，value就是label出现次数，因为数组从0开始，但计数从1
  #开始，故需要加1
  sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
  #返回一个列表按照第二个元素降序排列
  return sortedClassCount[0][0]#返回出现次数最多到label值，即为当前点的预测分类

def file2matrix(filename):#將文本记录转化为转化为Numpy矩阵
   fr=open(filename)#打开文件，存到fr里
   arrayOLines=fr.readlines()#按行读取，并存到arrOlines里
   numberOfLines=len(arrayOLines)#读取其行数
   returnMat=zeros((numberOfLines,3))#建立文本文件行数，3列的矩阵，以后整理的文件存在这里面
   classLabelVector=[]#建立一个单列矩阵，存储其类
   index=0#索引值先清0
   #按行读取文本，并依次给其贴标签
   for line in arrayOLines:
     line=line.strip()#將文本每一行首尾的空格去掉
     listFromLine=line.split('\t')#矩阵中，每遇到一个'\t',便依次將这一部分赋给一个元素
     returnMat[index,:]=listFromLine[0:3]#將每一行的前三个元素依次赋予之前预留矩阵空间
     #classLabelVector.append(int(float(listFromLine[-1])))
     #对于每行最后一列，按照其值的不同，来给单列矩阵赋值
     if(listFromLine[-1]=='largeDoses'):
       classLabelVector.append(3)
     elif listFromLine[-1]=='smallDoses':
       classLabelVector.append(2)
     elif listFromLine[-1]=='didntLike':
       classLabelVector.append(1)
     index+=1#每执行一次，便向下一行再循环
   return returnMat,classLabelVector#返回两个矩阵，一个是三个特征组成的特征矩阵，另一个为类矩阵

def autoNorm(dataSet):#对每个特征进行归一化处理
   minVals=dataSet.min(0)#取数据集最大值
   maxVals=dataSet.max(0)#取数据集最小值
   ranges=maxVals-minVals#取差值即为范围
   normDataSet=zeros(np.shape(dataSet))#建立一个新0矩阵，其行数列数与数据集一致，处理后数据存这里
   m=dataSet.shape[0]#读取数据集行数
   normDataSet=dataSet-np.tile(minVals,(m,1))#现有数据集减去最小值矩阵
   normDataSet=normDataSet/np.tile(ranges,(m,1))#归一化处理
   return normDataSet,ranges,minVals

def datingClassTest():#对数据进行测试
   filename="datingTestSet.txt"#载入数据
   hoRatio=0.1#提取数据为10%
   datingDataMat,datingLabels=file2matrix('datingTestSet.txt')#將文本数据转化为numpy矩阵
   normMat,ranges,minVals=autoNorm(datingDataMat)#数据归一化
   m=normMat.shape[0]#获取数据矩阵行数
   numTestVecs=int (m*hoRatio)#获取测试集数据行数
   errorCount=0.0#错误数现清0
   #获取测试集的错误率
   for i in range(numTestVecs):
     #执行之前的分类器，并將分类结果放在classifierResult里
     classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
     print('the classifier came back with:%d,the real answer is %d'%(classifierResult,datingLabels[i]))
     #如果分类器的结果与标签值不一致，则將错误数加一
     if(classifierResult!=datingLabels[i]):
        errorCount+=1
   print('the total error rate is %%' '%f'%(errorCount/float(numTestVecs)*100))#输出错误率

def classifyPerson():#约会对象评价预测
  resultList=['not at all','in small doses','in large doses']#输出的结果可能列表
  percentTats=float(input('percentage of time wpent playing video games?'))#首先需要键入花在游戏上时间比重
  ffmiles=float(input('frequent flier miles earned per year?'))#再键入每年飞行里程
  icecream=float(input('liters of ice cream consumed per year?'))#最后再键入在冰淇淋的消耗量
  datingDataMat,datingLabels=file2matrix('datingTestSet.txt')#把训练集文本数据转化为向量，便于后续处理
  normMat,ranges,minVals=autoNorm(datingDataMat)#几个关键数据量归一化，便于处理
  inArr=np.array([ffmiles,percentTats,icecream])#將刚刚键入的数组成预测集，便于带入现有模型预测
  classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)#將预测结果存储
  print('You will probably like this guy:',resultList[classifierResult-1])#给出预测结果

def img2vector(filename):#將图像转化为向量
  fr=open(filename)#打开文件
  returnVector=zeros((1,1024))#构建预存一维向量，大小之所以为1024是因为图片大小为32*32=1*1024
  #將每行图像均转化为一维向量
  for i in range(32):
    lineStr=fr.readline()#按行读入每行数据
    for j in range(32):
      returnVector[0,32*i+j]=int(lineStr[j])#將每行的每个数据依次存到一维向量中
  return returnVector#返回处理好的一维向量
 
def handwritingClassTest():
    hwLabels = []#测试集的标签矩阵
    trainingFileList = listdir('machinelearning/Ch02/trainingDigits')#返回trainingDigits目录下的文件名
    m = len(trainingFileList)#返回文件夹下文件的个数
    trainingMat = np.zeros((m, 1024))#初始化训练的Mat矩阵,测试集向量大小为训练数据个数*1024，即多少张图像，就有多少行，一行存一个图像
    for i in range(m): #从文件名中解析出训练集的类别标签
        fileNameStr = trainingFileList[i] #获得文件的名字
        classNumber = int(fileNameStr.split('_')[0])##第一个字符串存储标签，故取分离后的第一个元素，即相当于获取了该图像类别标签
        hwLabels.append(classNumber)#将获得的类别标签添加到hwLabels中
        trainingMat[i,:] = img2vector('machinelearning/Ch02/trainingDigits/%s' % (fileNameStr))#将每一个文件的1x1024数据存储到trainingMat中
    neigh = kNN(n_neighbors = 3, algorithm = 'auto')#构建kNN分类器,第一个参数表示近邻数为3，算法为权重均匀的算法
    neigh.fit(trainingMat, hwLabels)#拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    testFileList = listdir('machinelearning/Ch02/trainingDigits')#返回testDigits目录下的文件列表
    errorCount = 0.0#错误检测计数，初始值为0
    mTest = len(testFileList)#测试数据的数量
    for i in range(mTest):#从文件中解析出测试集的类别并进行分类测试
        fileNameStr = testFileList[i]#获得文件的名字
        classNumber = int(fileNameStr.split('_')[0])#获得分类的数字标签
        vectorUnderTest = img2vector('machinelearning/Ch02/trainingDigits/%s' % (fileNameStr)) #获得测试集的1x1024向量,用于训练
        classifierResult = neigh.predict(vectorUnderTest)#获得预测结果
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if(classifierResult != classNumber):#如果预测结果与实际结果不符，则错误数加一
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest * 100))#获取错误率
