
# coding: utf-8

# In[1]:
import numpy as np#导入numpy库
import random

def loadDataSet():#加载数据集函数
    dataMat=[]#数据集预留矩阵
    labelMat=[]#标签集预留矩阵
    fr=open('machinelearning/Ch05/testSet.txt')#打开数据集并存在fr里
    for line in fr.readlines():#按行读入
        lineArr=line.strip().split()#去回车载入数据
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#把每行的第一第二个数据作为数据集
        labelMat.append(int(lineArr[2]))#每行第三个作为标签
    return dataMat,labelMat

def sigmoid(inX):#激活函数
    return (1.0/(1.0+np.exp(-inX)))

def gradDesent(dataMatIn,classLabels):#梯度下降
    dataMatrix=np.mat(dataMatIn)#导入数据转成numpy格式矩阵
    labelMat=np.mat(classLabels).transpose()#将标签矩阵转置
    m,n=np.shape(dataMatrix)#将导入数据矩阵的长和宽附给m和n
    alpha=0.001#每次的步率为0.01
    maxCycles=50000#重复次数
    weights=np.ones((n,1))#权重初值矩阵
    for k in range(maxCycles):#循环执行梯度下降法
        h=sigmoid(dataMatrix*weights)#处理后为100*1矩阵
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error#这里不是直接求偏导，而是求导后的式子直接带入，此时3*1矩阵
    return weights.getA()

def plotBestFit(weights):#画图函数
    import matplotlib.pyplot as plt #载入matplotlib
    dataMat,labelMat=loadDataSet()#载入数据集和标签集
    dataArr=np.array(dataMat)#把数据转化为numpy矩阵
    n=np.shape(dataMat)[0]#获取第一列行数，即数据个数
    xcord1=[]#正样本
    ycord1=[]#正样本
    xcord2=[]#负样本
    ycord2=[]#负样本
    for i in range(n):
        if int(labelMat[i]==1):#正样本归类
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:#负样本归类
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)#添加subplot
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)#绘制正样本
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)#绘制负样本
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]#分界线,即sigmoid为0时的曲线应该为分界线，此时的x1与x2关系曲线
    ax.plot(x,y)#x,y绘在图纸上
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def stocGranfAscent0(dataMatrix,classLabels):#SGD函数
    m,n=np.shape(dataMatrix)#获取数据集的长宽
    alpha=0.01#每次的步率
    weights=np.ones(n)#权重矩阵，赋初值全为1
    for i in range(m):#做m次SGD
        h=sigmoid(sum(dataMatrix[i]*weights))#每一行的数据和做激活函数运算，即输出的类
        error=classLabels[i]-h#实际值与运算值做差值
        weights=weights+alpha*error*dataMatrix[i]#代入weights的迭代公式进行迭代
    return weights

def stocGranfAscent1(dataMatrix,classLabels,numIter=1500):#DSGD函数
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))#将所有m值列表化表示出来
        for i in range(m):            
            alpha=4/(1.0+j+i)+0.01#alpha为变值，随运算进行不断减少
            randIndex=int(random.uniform(0,len(dataIndex)))#随机从列表中取值，减少周期波动
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])#每计算一次权重更新后将刚刚用的m内的值删除，即不会再用 
    return weights
                   
def classifyVector(inX,weights):
                   prob=sigmoid(sum(inX*weights))
                   if prob>0.5: return 1.0
                   else: return 0
                   
def colicTest():
    frTrain=open('machinelearning/Ch05/horseColicTraining.txt')
    frTest=open('machinelearning/Ch05/horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGranfAscent1(np.array(trainingSet), trainingLabels, 50000)        #使用改进的随即上升梯度训练
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights))!= int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100                         #错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)
    return errorRate
                   
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iteration the zverage error rate is:                  %f"%(numTests,errorSum/float(numTests)))
                   
                  
