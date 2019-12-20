# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 12:22:26 2018

@author: Koustubh Vijay Kulkarni
"""
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import pandas as pd
import math

TrainingPercent = 80 #We are dividing our data set into training, validation and testing set. 80% for training, 10 % for validation adn 10 % for testing.
ValidationPercent = 10
TestPercent = 10


def generatetarget(humanConcat_data):
    t=[]
    t=humanConcat_data["target"]
    return t
def generateData(humanConcat_data,DSvalue):
    dataMatrix = [] 
    if(DSvalue==1):
        dataMatrix = humanConcat_data.filter(['f1_x','f2_x','f3_x','f4_x','f5_x','f6_x','f7_x','f8_x','f9_x','f1_y','f2_y','f3_y','f4_y','f5_y','f6_y','f7_y','f8_y','f9_y'], axis=1)
    if(DSvalue==2):
        dataMatrix = humanConcat_data.filter(['f1','f2','f3','f4','f5','f6','f7','f8','f9'], axis=1)
    if(DSvalue==3):
        dataMatrix = humanConcat_data.filter(regex=("f.*"), axis=1)
    if(DSvalue==4):
        dataMatrix = humanConcat_data.filter(regex=("f.*"), axis=1)
    return dataMatrix

def getTrainingDataSet(rawData, targetVector,TrainingPercent):
    totalrows=len(rawData.index)
    Training_rows = math.ceil(totalrows*0.01*TrainingPercent)
    return rawData.head(Training_rows),targetVector.head(Training_rows)

def getTestingDataSet(rawData,targetVector, TestPercent,TrainingCount,ValCount):
    totalrows=len(rawData.index)
    test_rows = math.ceil(totalrows*0.01*TestPercent)
    End_test_rows=TrainingCount+ValCount+test_rows
    return rawData[TrainingCount+ValCount:End_test_rows],targetVector[TrainingCount+ValCount:End_test_rows]

def getValDataSet(rawData,targetVector, ValPercent, TrainingCount):
    totalrows=len(rawData.index)
    Val_rows = math.ceil(totalrows*0.01*ValPercent)
    End_val_rows=TrainingCount+Val_rows
    return rawData[TrainingCount:End_val_rows], targetVector[TrainingCount:End_val_rows]

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,DSvalue):
    DataT       = np.transpose(Data)
    BigSigma    = np.zeros((len(DataT),len(DataT)))#Return a new array of given shape and type, filled with zeros.
    TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))        
    varVect     = []
    print("Big SIgma shape",BigSigma.shape)
    if (DSvalue==1 or DSvalue==2):        
        for i in range(1,10):
            if (DSvalue==1 or DSvalue==3):
                varVect.append(np.var(Data["f%d_x" %i]))
            if (DSvalue==2 or DSvalue==4):
                varVect.append(np.var(Data["f%d" %i]))
        
        for i in range(1,10):
            if (DSvalue==1):
                varVect.append(np.var(Data["f%d_y" %i]))
    elif (DSvalue==3 or DSvalue==4):
        for i in range(1,513):
            if (DSvalue==3):
                varVect.append(np.var(Data["f%d_x" %i]))
            if (DSvalue==4):
                varVect.append(np.var(Data["f%d" %i]))
        
        for i in range(1,513):
            if (DSvalue==3):
                varVect.append(np.var(Data["f%d_y" %i]))
    #np.var-Returns the variance of the array elements, a measure of the spread of a distribution.
    for j in range(len(DataT.index)):
        BigSigma[j][j] = varVect[j]+0.0001
    BigSigma = np.dot(200,BigSigma)
    return BigSigma
#This method returns the scalar value for each basis function for each datapoint. These values will ultimately be part of our design matrix.
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow) # Represents (X-Mu) term in radial basis fucntion formula
    T = np.dot(BigSigInv,np.transpose(R)) # We are taking transpose of BIgSigma matrix that we generated and taking dot prodcut with (X-Mu)  
    L = np.dot(R,T)
    return L
#This method computes radial basis function for respictive data and Mu. It basically forms a design matrix. 
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x
#This method creates design matrix.
def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(Data)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    #BigSigma=np.add(BigSigma,0.005)
    BigSigInv = np.linalg.inv(BigSigma) #computes the inverse of our BigSigma matrix
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(Data.iloc[R], MuMatrix[C], BigSigInv)
    return PHI
def GetValTest(VAL_PHI,W):#here we calculate predicted output which we get from transpose of our design matrix and weight matrix.
    Y = np.dot(W,np.transpose(VAL_PHI))
    return Y
#This method computes ERMS for our training,validation and testing data. Our model is behavinf correct if the ERMS we get is low.
def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    accuracy = 0.0
    counter = 0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct.iloc[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct.iloc[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def Crossloss(h, y,m):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h))/m

def CalculateLinearGD(humanConcat_data,datasetname,DSvalue):
    if(DSvalue==1 or DSvalue==3):
        M=13
    elif(DSvalue==2 or DSvalue==4):
        M=9
    W_Now        = np.random.rand(M)*0 #In SGD, we are initalising our weights by scaling weights we got from closed form. Here I am initialising by o
    La           = 2 # This is regularization parameter
    learningRate = 0.1 #This is learning rate which decides how fast algo will converge.
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    #I have defined three new arrays to store accuracy.
    L_Accu_Val   = []
    L_Accu_TR    = []
    L_Accu_Test  = []
    W_Mat        = []
    #Get Target Data and feature data
    targetVector=generatetarget(humanConcat_data)
    targetData=generateData(humanConcat_data,DSvalue)
    print("targetVector shape",targetVector.shape)
    print("targetData shape",targetData.shape)
    
    trainingData,trainingTarget=getTrainingDataSet(targetData,targetVector, TrainingPercent)
    print("trainingData shape",trainingData.shape)
    print("trainingTarget shape",trainingTarget.shape)
    
    valData,valTarget=getValDataSet(targetData,targetVector, ValidationPercent,len(trainingData))
    print("validationData shape",valData.shape)
    print("valTarget shape",valTarget.shape)
    
    testingData,testingTarget=getTestingDataSet(targetData,targetVector, TestPercent,len(trainingData),len(valData))
    print("testingData shape",testingData.shape)
    print("testingTarget shape",testingTarget.shape)
    
    kmeans = KMeans(n_clusters=M, random_state=0).fit(trainingData)
    Mu = kmeans.cluster_centers_
    BigSigma = GenerateBigSigma(targetData, Mu, TrainingPercent,DSvalue)
    TRAINING_PHI = GetPhiMatrix(targetData, Mu, BigSigma, TrainingPercent) #Design matrix for training
    TEST_PHI     = GetPhiMatrix(testingData, Mu, BigSigma, 100) #Design matrix for testing data
    VAL_PHI      = GetPhiMatrix(valData, Mu, BigSigma, 100) #Design matrix for validation data
    
    print(Mu.shape)
    print("Big SIgma shape",BigSigma.shape)
    print(TRAINING_PHI.shape)
    
    for i in range(0,1000):
        
        #Here we are calculating delta_ED,Delta E and Delta W. once we get these values we are calcuating new weights. In next few lines we are simply implementing formulas for these computations.
        Delta_E_D     = -np.dot((trainingTarget.iloc[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
        La_Delta_E_W  = np.dot(La,W_Now)
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
        Delta_W       = -np.dot(learningRate,Delta_E)
        W_T_Next      = W_Now + Delta_W
        W_Now         = W_T_Next
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,trainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        L_Accu_TR.append(float(Erms_TR.split(',')[0]))
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,valTarget)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        L_Accu_Val.append(float(Erms_Val.split(',')[0]))
        
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,testingTarget)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        L_Accu_Test.append(float(Erms_Test.split(',')[0]))
    
    
    # In[ ]:
    print ('UBITname      = Kkulkarn')
    print ('Person Number = 50288207')
    print ('----------Linear Regression Gradient Descent Solution '+datasetname+'--------------------')
    print ("M ="+str(M)+" \nLambda  = "+str(La)+"\neta=" +str(learningRate))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))
    print ("Accuracy Training   = " + str(np.around(max(L_Accu_TR),5)))
    print ("Accuracy Validation = " + str(np.around(max(L_Accu_Val),5)))
    print ("Accuracy Testing    = " + str(np.around(max(L_Accu_Test),5)))
    
humanConcat_data=pd.DataFrame()
humanConcat_data = pd.read_csv('HumanConcat.csv')
#CalculateLinearGD(humanConcat_data,'For Human Concatenated Dataset',1)
humanConcat_data = pd.read_csv('HumanSub.csv')
#CalculateLinearGD(humanConcat_data,'For Human Subtracted Dataset',2)
humanConcat_data = pd.read_csv('GSCConcat.csv')
#CalculateLinearGD(humanConcat_data,'For GSC Concatenated Dataset',3)
humanConcat_data = pd.read_csv('GSCSub.csv')
#CalculateLinearGD(humanConcat_data,'For GSC Subtracted Dataset',4)

################ LOGISTIC REGRESSION ###################################################
def predict(final_pred, m):
    y_pred = np.zeros((m,1))
    print("pred-->",final_pred.shape[0])
    for i in range(final_pred.shape[0]):
        if final_pred[i] > 0.5:
            y_pred[i] = 1
    return y_pred

def CalculateLogisticGD(humanConcat_data,datasetname,DSvalue):
   totalloss=[]
   targetVector=generatetarget(humanConcat_data)
   targetData=generateData(humanConcat_data,DSvalue)
   print("targetVector shape",targetVector.shape)
   print("targetData shape",targetData.shape)
   
   W_Now        = np.random.rand(len(np.transpose(targetData)))*0
   trainingData,trainingTarget=getTrainingDataSet(targetData,targetVector, TrainingPercent)
   print("trainingData shape",trainingData.shape)
   print("trainingTarget shape",trainingTarget.shape)
   valData,valTarget=getValDataSet(targetData,targetVector, ValidationPercent,len(trainingData))
   print("validationData shape",valData.shape)
   print("valTarget shape",valTarget.shape)
    
   testingData,testingTarget=getTestingDataSet(targetData,targetVector, TestPercent,len(trainingData),len(valData))
   print("testingData shape",testingData.shape)
   print("testingTarget shape",testingTarget.shape)
   
   W_new,totalloss = getLogisticResult(trainingData,trainingTarget,W_Now,len(trainingData.index)) 
   
   training_Output=sigmoid(np.dot(W_new,np.transpose(trainingData)))
   train_pred=predict(training_Output,len(trainingData.index))
   
   val_Output=sigmoid(np.dot(W_new,np.transpose(valData)))
   val_pred=predict(val_Output,len(valData.index))
   valLoss   = Crossloss(val_Output,valTarget,len(valTarget))
   
   testing_Output=sigmoid(np.dot(W_new,np.transpose(testingData)))
   test_pred=predict(testing_Output,len(testingData.index))
   testLoss   = Crossloss(testing_Output,testingTarget,len(testingTarget))
   
   print ('UBITname      = Kkulkarn')
   print ('Person Number = 50288207')
   print ('----------Logistic Regression result for '+datasetname+'--------------------')
   print ("Accuracy Training   = " + str(accuracy_score(train_pred, trainingTarget)*100))
   print ("Accuracy validation   = " + str(accuracy_score(val_pred, valTarget)*100))
   print ("Accuracy Testing   = " + str(accuracy_score(test_pred, testingTarget)*100))
   
   
   ErmsArr = []
   AccuracyArr = []

def getLogisticResult(targetData,targetVector,W_Now,m):
   loss=[]
   learningRate = 0.03 
   for i in range(0,6500):
       z = np.dot(targetData, np.transpose(W_Now))
       h = sigmoid(z)
       Delta_W = np.dot(np.transpose(targetData), (h - targetVector))/ m
       Itrloss = (-1/m)*(np.sum((targetVector*np.log(h)) + ((1-targetVector)*(np.log(1-h)))))
       W_Now    = W_Now - (np.dot(learningRate,np.transpose(Delta_W)))
        
       loss.append(Itrloss)
   return W_Now, loss
    
humanConcat_data=pd.DataFrame()
humanConcat_data = pd.read_csv('HumanConcat.csv')
#CalculateLogisticGD(humanConcat_data,'For Human Concatenated Dataset',1)
humanConcat_data = pd.read_csv('HumanSub.csv')
#CalculateLogisticGD(humanConcat_data,'For Human Subtracted Dataset',2)
humanConcat_data = pd.read_csv('GSCConcat.csv')
#CalculateLogisticGD(humanConcat_data,'For GSC Concatenated Dataset',3)
humanConcat_data = pd.read_csv('GSCSub.csv')
#CalculateLogisticGD(humanConcat_data,'For GSC Subtracted Dataset',4)


##################################### Neural Network Code #############################

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from pandas_ml import ConfusionMatrix

import numpy as np

def get_model(input_size):
    drop_out = 0.1
    first_dense_layer_nodes  = 900
    second_dense_layer_nodes = 1
    
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu'))
    
    """In this model, we are using 'Reactified Linear Unit'(Relu) as an activation function. Relu simply activates nodes which has positive value.
    Relu is famous because it does not cause vanishing gradient problem.
    """
    
    # Why dropout?-To prevent overfitting , depending on the value we set as dropout -some nodes are randomly not used while training the model.
    model.add(Dropout(drop_out))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    
    
    model.summary()#Print the summary representation of your model
    
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
def getNeuralResult(dataset,datasetname,DSvalue):
    
    validation_data_split = 0.10 #Spliting of training set in training and validation data.
    num_epochs = 5000 #Maximum no of epochs
    model_batch_size = 200 #Batch size indicates no of samples to be included in single batch
    tb_batch_size = 32
    early_patience = 200 #Stops when the loss remains contant or stops showing any improvement i.e. for e.g. if this value is set to 100 then if after 100 epochs loss remains same then it will stop the training. 
    
    targetVector=generatetarget(dataset)
    targetData=generateData(dataset,DSvalue)
    trainingData,trainingTarget=getTrainingDataSet(targetData,targetVector, 90)
    testingData,testingTarget=getTestingDataSet(targetData,targetVector, TestPercent,len(trainingData),0)
    input_size = len(np.transpose(trainingData)) 
    print("input_size",input_size)
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    
    # Read Dataset
  
    model = get_model(input_size)
    # Process Dataset 
    history = model.fit(trainingData
                        , trainingTarget
                        , validation_split=validation_data_split
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , callbacks = [tensorboard_cb,earlystopping_cb]
                       )
    
    
    df = pd.DataFrame(history.history)
    df.plot(subplots=True, grid=True, figsize=(10,15))
    
    
    print("testingData-->",testingData.shape)
    acc_Score=model.evaluate(testingData,testingTarget)
    print(acc_Score)
    print ('UBITname      = Kkulkarn')
    print ('Person Number = 50288207')
    print("NN Results for "+datasetname)
    print('Loss:',float(acc_Score[0]))
    print('Accuracy:',float(acc_Score[1]*100))
    
humanConcat_data=pd.DataFrame()
humanConcat_data = pd.read_csv('HumanConcat.csv')
getNeuralResult(humanConcat_data,'For Human Concatenated Dataset',1)
humanConcat_data = pd.read_csv('HumanSub.csv')
getNeuralResult(humanConcat_data,'For Human Subtracted Dataset',2)
humanConcat_data = pd.read_csv('GSCConcat.csv')
getNeuralResult(humanConcat_data,'For GSC Concatenated Dataset',3)
humanConcat_data = pd.read_csv('GSCSub.csv')
getNeuralResult(humanConcat_data,'For GSC Subtracted Dataset',4)