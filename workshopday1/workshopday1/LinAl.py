# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:52:53 2017

@author: Dr.Srinivas
"""
import numpy as np
import numpy.linalg as lp


v = [-1.0,3.0,4.0]

A = [[1.0,2.0],[3.0,4.0]]

C = [[1.0,1.5,-2.0],[2.0,1.0,-1.0],[3.0,-1.0,2.0]]

B = [[3.0,-2.0],[2.0,1.0]]


print([[A[i][j]+B[i][j] for j in range(len(B[0]))] for i in range(len(A))]) 


print(sum(v[i]*v[i] for i in range(len(v))))


#Matrix Multiplication

print([[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))] )





C = [[1.0,1.5,-2.0],[2.0,1.0,-1.0],[3.0,-1.0,2.0]]


#Inverse
Ainv = lp.inv(C)

print (Ainv)

#lp.solve(C,D)
lp.solve(Ainv,v)


A = [[1.0,2.0],[3.0,4.0]]

C = [[1.0,1.5,-2.0],[2.0,1.0,-1.0],[3.0,-1.0,2.0]]

B = [[3.0,-2.0],[2.0,1.0]]


a = np.matrix(A)
b = np.matrix(B)
c = a+b
d = a*b

print([[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))] )

#inverse times a matrix is Identity
np.matrix(C)*np.matrix(lp.inv(C))
#solve 2 x + 3 y = 14, 3 x + 2 y = 16 
#A y = b
#y = inv(A) *b

np.matrix(a)*np.matrix(lp.inv(a))

A = [[1.0,2.0],[3.0,4.0]]

C = [[1.0,1.5,-2.0],[2.0,1.0,-1.0],[3.0,-1.0,2.0]]

B = [[3.0,-2.0],[2.0,1.0]]


a = np.matrix(A)
b = np.matrix(B)

lp.eigvals(a)

lp.eigvals(C)

lp.eig(a)

lp.eig(C)

E= [[1.1,2.2,3.1],[1.3,2.5,4.3]]
e=np.matrix(E)

print(a*e)

f=a-b
lp.vdot(v,v)




#lp mat mult

import scipy

scipy.dot(a,b)
scipy.dot(np.matrix(C),np.matrix(Ainv))  #multiplying a matrix with inverse is identity

