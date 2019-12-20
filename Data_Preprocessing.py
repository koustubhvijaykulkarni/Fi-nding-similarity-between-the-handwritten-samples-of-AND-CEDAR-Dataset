"""
Created on Fri Oct 26 12:22:26 2018

@author: Koustubh Vijay Kulkarni
"""
import numpy as np
import pandas as pd
from random import sample

def processData(dataset):
    Sample_1   = dataset['img_id_A'].values
    Sample_2 = dataset['img_id_B'].values
    target = dataset['target'].values
    
    return Sample_1, Sample_2, target

samePairs_dataset = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\same_pairs.csv')
print("X shape is", samePairs_dataset.shape, "\nY shape is", samePairs_dataset.shape)
processedData1,processedData2, processedLabel = processData(samePairs_dataset)
print("processedLabel -", processedLabel.shape)

diffnPairs_dataset = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\diffn_pairs.csv')
print("X1 shape is", diffnPairs_dataset.shape, "\nY1 shape is", diffnPairs_dataset.shape)
diffnPairs_dataset=diffnPairs_dataset.sample(791)
print("diffnPairs_dataset -", diffnPairs_dataset.shape)
processedData11,processedData22, processedLabel2 = processData(diffnPairs_dataset)

column1=np.concatenate((processedData1, processedData11), axis=0)
column2=np.concatenate((processedData2, processedData22), axis=0)
column3=np.concatenate((processedLabel, processedLabel2), axis=0)

dataset = {}
dataset["img_id_A"]  = column1
dataset["img_id_B"] = column2
dataset["target"] = column3
dataset=pd.DataFrame(dataset)
pd.DataFrame(dataset).to_csv('samples.csv')    

newdata=pd.DataFrame()
humanObsrvd_dataset = pd.read_csv('HumanObserved-Dataset\HumanObserved-Features-Data\HumanObserved-Features-Data.csv')
humanObsrvd_dataset=pd.DataFrame(humanObsrvd_dataset)
newdata=pd.merge(dataset,humanObsrvd_dataset,how='inner',left_on="img_id_A", right_on="img_id")
newdata=newdata.drop(columns=['img_id', 'Unnamed: 0']) 
newdata=pd.merge(newdata,humanObsrvd_dataset,how='inner',left_on="img_id_B", right_on="img_id")
newdata=newdata.drop(columns=['img_id', 'Unnamed: 0']) 
newdata = newdata.sample(frac=1, random_state=300).reset_index(drop = True)
pd.DataFrame(newdata).to_csv('HumanConcat.csv')

subdata=pd.DataFrame()
subdata["img_id_A"]  = column1
subdata["img_id_B"] = column2
subdata["target"] = column3
for i in range(1, 10):
    subdata["f%d" %i]=newdata["f%d_x" %i]-newdata["f%d_y" %i]

subdata = subdata.sample(frac=1, random_state=300).reset_index(drop = True)
pd.DataFrame(subdata).to_csv('HumanSub.csv')


######################################################GSC#############################################




samePairs_datasetGSC = pd.read_csv('GSC-Dataset\GSC-Features-Data\same_pairs.csv')
print("X shape is", samePairs_datasetGSC.shape, "\nY shape is", samePairs_datasetGSC.shape)

samePairs_datasetGSC=samePairs_datasetGSC.sample(4000)
processedData1_GSC,processedData2_GSC, processedLabel_GSC = processData(samePairs_datasetGSC)
print("processedLabel_GSC -", processedLabel_GSC.shape)

diffnPairs_dataset_GSC = pd.read_csv('GSC-Dataset\GSC-Features-Data\diffn_pairs.csv')
print("X1 shape is", diffnPairs_dataset_GSC.shape, "\nY1 shape is", diffnPairs_dataset_GSC.shape)

diffnPairs_dataset_GSC=diffnPairs_dataset_GSC.sample(4000)
print("diffnPairs_dataset_GSC -", diffnPairs_dataset_GSC.shape)
processedData11_GSC,processedData22_GSC, processedLabel2_GSC = processData(diffnPairs_dataset_GSC)

column1_GSC=np.concatenate((processedData1_GSC, processedData11_GSC), axis=0)
column2_GSC=np.concatenate((processedData2_GSC, processedData22_GSC), axis=0)
column3_GSC=np.concatenate((processedLabel_GSC, processedLabel2_GSC), axis=0)

dataset_GSC = {}
dataset_GSC["img_id_A"]  = column1_GSC
dataset_GSC["img_id_B"] = column2_GSC
dataset_GSC["target"] = column3_GSC
dataset_GSC=pd.DataFrame(dataset_GSC)

dataset_GSC = dataset_GSC.sample(frac=1, random_state=300).reset_index(drop = True)
pd.DataFrame(dataset_GSC).to_csv('samples_GSC.csv')    
subdata_GSC=pd.DataFrame()
subdata_GSC=dataset_GSC

newdata_GSC=pd.DataFrame()
humanObsrvd_dataset_GSC = pd.read_csv('GSC-Dataset\GSC-Features-Data\GSC-Features.csv')
humanObsrvd_dataset_GSC=pd.DataFrame(humanObsrvd_dataset_GSC)
newdata_GSC=pd.merge(dataset_GSC,humanObsrvd_dataset_GSC,how='inner',left_on="img_id_A", right_on="img_id")
newdata_GSC=pd.merge(newdata_GSC,humanObsrvd_dataset_GSC,how='inner',left_on="img_id_B", right_on="img_id")
newdata_GSC=newdata_GSC.drop(columns=['img_id_x', 'img_id_y']) 
newdata_GSC = newdata_GSC.sample(frac=1, random_state=300).reset_index(drop = True)
pd.DataFrame(newdata_GSC).to_csv('GSCConcat.csv')


for i in range(1, 513):
    subdata_GSC["f%d" %i]=newdata_GSC["f%d_x" %i]-newdata_GSC["f%d_y" %i]

subdata_GSC = subdata_GSC.sample(frac=1, random_state=300).reset_index(drop = True)
pd.DataFrame(subdata_GSC).to_csv('GSCSub.csv')