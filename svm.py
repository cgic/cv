# --------------scikit-learn library---------------
# classifier that uses
# geographical data to predict the type of forest in an area.
# You need to submit separate 2 files (don¡¯t bundle them in a zip file submit the two
# files separately): (1) a solutions file, and (2) the Python code for you classifier.

# 1. The solutions file:
# a. Naming convention: This file should be named using the following convention
# your solution file ¡®studentnumber.txt¡¯, e.g. C1234567.txt.

# Contents: The file should list your classifier¡¯s target variable predictions for
# each of the query instances in the queries.txt file. Each line in the file should
# list one query id followed by a comma followed by your classifier¡¯s prediction
# for that query, i.e.:
# <tstid>,<prediction>
# The box below illustrates what someone looking at a portion of your solutions
# files should see

# tst100,<=50K
# tst101,>50K
# tst102,<=50K
# tst103,<=50K
# tst104, >50K

# The Python code for your classifier.
# a. Naming convention: This file should be named using the following
# convention ¡®studentnumber.py¡¯ .

# b. Contents: This code should expect the training and query data to be in a
# subdirectory of the directory its in called ¡®data¡¯. It should have a main
# function which when run creates your solution file and stores it in a
# subdirectory called ¡®solutions¡¯. Make sure to include your names and
# student numbers as comments at the top of you python code file. Your
# code should be appropriately commented.
# Marking Scheme
# Marks are awarded based on the accuracy of the classifier. The accuracy metric used
# will be the average class accuracy (harmonic mean) of the classifier.

import numpy as np
from sklearn import svm
import os, sys, glob, cv2, math
if sys.version_info[0] < 3: range = xrange

# rootPath = os.path.dirname(__file__)

# traindata = os.path.join(rootPath,'trainingset.txt\\trainingset.txt')
# querydata = os.path.join(rootPath,'queries.txt\\queries.txt')

def maxminstandard(x):
    x = (x - np.max(x,axis=0)) / (np.max(x,axis=0) - np.min(x,axis=0))
def normalstandard(x):
    x = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    



class SVM:
    def __init__(self,X=None,y=None,mean=None,std=None,max=None,min=None):
        self.clf = svm.NuSVC()
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std
        self.max = max
        self.min = min
    def train(self,traindata,delim=',',process=normalstandard):
        self.X = np.array([[float(v) for v in line.split(delim)[1:-1]] for line in open(traindata)], dtype = np.float32)
        self.mean = self.X.mean(axis = 0)
        self.std = self.X.std(axis = 0)
        self.max = self.X.max(axis = 0)
        self.min = self.X.min(axis = 0)
        process(self.X)
        self.y = np.array([line.rstrip().split(delim)[-1] for line in open(traindata)])
        self.clf.fit(self.X,self.y)
    def predict(self,data):
        return self.clf.predict(data)
    def predictAnswer(self,testdata,output='pre.answer',delim=','):
        predata = np.array([[float(v) for v in line.rstrip().split(delim)[1:-1]] for line in open(testdata)], dtype = np.float32)
        prename = np.array([line.split(delim)[0] for line in open(testdata)])
        predata = (predata-self.mean)/self.std
        prediction = self.clf.predict(predata)
        print prediction
        f = open(output,'w')
        content = ''
        for i,v in enumerate(prediction):
            content += prename[i] + delim + v + '\n'
        f.write(content)
        f.close()

def getImageFeature(image):
    im = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2HSV)
    size = im.shape[0],im.shape[1]
    partsize = size[0]/3, size[1]/3
    feature = ()
    for part in range(9):        
        cropsize = partsize[0]*(part%3),partsize[1]*(part/3), partsize[0]*(part%3)+partsize[0],partsize[1]*(part/3)+partsize[1]
        impart = im[cropsize[0]:cropsize[2],cropsize[1]:cropsize[3]]
        mean = (np.mean(im[:,:,0]),np.mean(im[:,:,1]),np.mean(im[:,:,2]))
        variance = (np.var(im[:,:,0]),np.var(im[:,:,1]),np.var(im[:,:,2]))
        skewness = (np.mean((im[:,:,0]-mean[0])**3) / math.pow(variance[0],1.5),
                    np.mean((im[:,:,1]-mean[1])**3) / math.pow(variance[1],1.5),
                    np.mean((im[:,:,2]-mean[2])**3) / math.pow(variance[2],1.5))
        feature += mean+variance+skewness        
    return feature      
    
#type=1: generate train data from images
#type=0: generate test data from images
def saveImageFeatures(pathlist,delim=',',prefix='tr',save='train.dat',type=1):
    imlist = []
    f = open(save,'w')
    for path in pathlist:
        imlist += [(img,path) for img in glob.glob(path + '\\*.jpg')]   
    for k,v in enumerate(imlist):
        if type:
            prefix = prefix+str(k)+delim
        else:
            prefix = v[0].split('\\')[-1] + delim
        f.write(prefix)
        feature = getImageFeature(v[0])
        for i in range(81):
            f.write(str(feature[i]) + delim)
        f.write('%s\n' % (v[1] if type else '?'))
    f.close()
            
if __name__ == '__main__':
    # filePathList = ['beach','grassland']
    # saveImageFeatures(filePathList)
    # saveImageFeatures(['tst'],save='pre.dat',type=0)
    st = SVM()
    st.train('train.dat')
    # print st.clf.intercept_,st.clf.support_vectors_,st.clf.support_
    st.predictAnswer('pre.dat')
