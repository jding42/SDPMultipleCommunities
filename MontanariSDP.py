#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:30:13 2019

@author: jingqiu
"""

import numpy as np
import scipy as sci
from sklearn.preprocessing import normalize
import numpy.random as random
from sklearn.decomposition import PCA
import itertools
import numpy.linalg as lin
from multiprocessing import Pool

def MontanariSDP(X,m,epochs,d,epsilon=1E-3):
    #initialize
    n=len(X)
    #d=sum([len(x) for x in X])/n
    print(d)
    Y=random.choice([1,-1],(n,m))
    Y=normalize(Y)
    for epoch in range(epochs):
        print(epoch)       
        flag=0
        M=np.sum(Y,axis=0)
        #for i in random.permutation(range(n)):
        yold=np.zeros(m)
        #for i in random.permutation(range(n)):
        maxdiff=0.0
        for i in random.choice(n,n):
            Ni=X[i]
            #print(Ni)
            #M+=np.sum(Y,axis=0)
            #Y_new[i,:]=np.sum(Y[Ni,:],axis=0)-M*(d)/n 
            if(len(Ni)>0):
                yold[:]=Y[i,:]
                Y[i,:]=np.sum(Y[Ni,:],axis=0)-M*d/n
                Y[i,:]/=lin.norm(Y[i,:])
                M+=Y[i,:]-yold
                maxdiff=max(maxdiff,lin.norm(yold-Y[i,:]))
        print(lin.norm(M))
        print(maxdiff)
        #Y=normalize(Y_new)    
        if(maxdiff<epsilon):
            break
    sigma=Y.T@Y/n
    #print(sigma)
    #pca=PCA(n_components=1, svd_solver='arpack')
    #pca.fit(Y)
    u,s,vh=lin.svd(sigma,full_matrices=True)
    #print(pca.components_[0,:])
    return (Y.dot(u[:,0])>0)*1

def SBM(n,d,epsilon,k=2):
    Y=random.randint(k,size=n)
    #print(Y)
    X=[[] for i in range(n)]
    random_num=random.rand((n*(n-1))//2);
    t=0
    for i in range(n):
        for j in range(i):
            if(Y[i]==Y[j]):
                p=(1+(1-1/k)*epsilon)*d/n
            else:
                p=(1-epsilon/k)*d/n
            if(random_num[t]<p):
                X[i].append(j)
                X[j].append(i)
            t+=1
    return X,Y

def max_correlation(Yhat,Y,k):
    return max([np.sum(np.array(label_permu)[Yhat]==Y)/np.size(Yhat) for label_permu in list(itertools.permutations(range(k)))])
 
def MultiCommunitySDP(X,k,m,epochs,d,epsilon=1E-3,gamma=1.0,threshold=True):
    n=len(X)
    #d=sum([len(x) for x in X])/n
    #print(d)
    y=random.choice(range(k),n*m)
    Y=np.zeros((n,k*m))
    for i in range(n):
        for j in range(m):
            Y[i,y[i*m+j]+j*k]=1
    #Y=random.choice([0,-1/k],(n,k*m))
    #Y=normalize(Y)
    #print(np.sum(Y,axis=1))
    #print(np.sum(Y,axis=0))
    epoch=0
    while True:
    #for epoch in range(epochs):
        #print(epoch)       
        flag=0
        M=np.sum(Y,axis=0)
        #print(M)
        #print(M)
        #if(epoch==0):
        #if(epoch%100==0):
        #    print(lin.norm(M-n/(k)))
        #for i in random.permutation(range(n)):
        yold=np.zeros(m*k)
        #for i in random.permutation(range(n)):
        maxdiff=0.0
        for i in random.choice(n,n):
            Ni=X[i]
            #print(Ni)
            #M+=np.sum(Y,axis=0)
            #Y_new[i,:]=np.sum(Y[Ni,:],axis=0)-M*(d)/n 
            if(len(Ni)>0):
                yold[:]=Y[i,:]
                #Y[i,:]=np.sum(Y[Ni,:],axis=0)-(M-n/(k*(m**0.5)))*gamma
                Y[i,:]=np.sum(Y[Ni,:],axis=0)-(M-n/(k))*gamma
                #Y[i,:]/=lin.norm(Y[i,:])/(m**(1/2))
                if(threshold):
                    Y[i,:]=np.maximum(0,Y[i,:])
                #normi=sum([np.sum(Y[i,j*k:(j+1)*k])**2 for j in range(m)])
                #Y[i,:]/=(normi**(1/2))
                Y[i,:]/=lin.norm(Y[i,:])/(m**(1/2))
                M+=Y[i,:]-yold
                maxdiff=max(maxdiff,lin.norm(yold-Y[i,:]))
        #if(epoch%100==0):
        #    print(maxdiff)
        #Y=normalize(Y_new)   
        epoch+=1
        if(maxdiff<epsilon and epoch>epochs):
            break
    #for i in range(k):
    Y-=1/(k)
    #print(Y[0,:])
    sigma=Y.T@Y/n
    #print(sigma)
    #pca=PCA(n_components=1, svd_solver='arpack')
    #pca.fit(Y)
    u,s,vh=lin.svd(sigma,full_matrices=True)
    #print(pca.components_[0,:])
    result=np.zeros((n,k))
    for i in range(k):
        result[:,i]=Y[:,i:k*m:k].dot(u[i:k*m:k,0])
    return result                  



def process_l(l):
    k=6
    d=3
    a=[]
    #b=[]
    n=4000
    iterations=4000
    rank=40
    record=[]
    for t in range(5):
        lamda=l*l
        epsilon=(lamda/d)**(1/2)*k
        X,Y=SBM(n,d,epsilon,k)
        #Yhat=MontanariSDP(X,rank,iterations,d,1E-3)
        Yhat=MultiCommunitySDP(X,k,rank,iterations,d,1E-2,threshold=True)
        Yhat=np.argmax(Yhat,axis=1)
        record.append(max_correlation(Yhat,Y,k))
    return record

#for l in [0.8,0.9,1.0,1.1,1.2]:
#for l in [0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5]:
#    record=[]
#    for t in range(5):
#        lamda=l*l
#        epsilon=(lamda/d)**(1/2)*k
#        X,Y=SBM(n,d,epsilon,k)
#        #Yhat=MontanariSDP(X,rank,iterations,d,1E-3)
#        Yhat=MultiCommunitySDP(X,k,rank,iterations,d,1E-3,threshold=False)
#        Yhat=np.argmax(Yhat,axis=1)
#        record.append(max_correlation(Yhat,Y,k))
#        #print(np.sum(Yhat==Y))
#    a.append(record)
#    print(record)
    
    #b.append(max([np.sum(Y==i) for i in range(k)]))

if __name__ == '__main__':
    with Pool(8) as p:
        print(p.map(process_l,[0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5]))
    
    
    