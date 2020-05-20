#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import sympy as sy
from sympy import *
init_printing(use_unicode=True)
A=np.mat("1 0.5 0;0 0.5 1;0 0 0")
X=np.mat("0.33;0.33;0.33")
print("the initial distributions of the plants are\n",X)
print("A:\n",A)
A.shape


# In[2]:


print("diagonalization of A is accomplished by finding eigen values and eigen vectors of A\n")
print("Eigenvalues:\n",np.linalg.eigvals(A))
print("lambda1:\n\n",(np.linalg.eigvals(A))[0])
print("lambda2:\n\n",(np.linalg.eigvals(A))[1])
print("lambda3:\n\n",(np.linalg.eigvals(A))[2])
lambda1=(np.linalg.eigvals(A))[0]


# In[3]:


eigenvalue,eigenvector=np.linalg.eig(A)
print("first eigen values lambda1 lambda2 and lambda3 are:",eigenvalue)
print(eigenvector)
print("first eigen vector V1 of A is\n\n",eigenvector[0])
print("first eigen vector V1 of A is\n\n",eigenvector[1])
print("first eigen vector V1 of A is\n\n",eigenvector[2])


# In[4]:


print("eigen vectors V1 V2 and V3 are:\n",eigenvector)
print("V1: \n \n",eigenvector[0].transpose())
print("V2: \n \n",eigenvector[1].transpose())
print("V3: \n \n",eigenvector[2].transpose())
eigenvector[0].transpose().shape


# In[5]:


x=eigenvector.transpose()
print(" vector matrix P consisting of the Eigen vectors  is\n",x)
x.shape


# In[7]:



#in this section we are sorting the eigen values in decending order and assigning it to a variable called eigen value
#we are also finding the corresponding eigen vectors

idx = eigenvalue.argsort()[::-1] 
eigenvalue = eigenvalue[idx]
eigenvector = eigenvector[:,idx]
#print("eigen vectors V1 V2 and V3 are:\n",eigenvector)
print("\n \n")
print("P is")
print(eigenvector.transpose())
print("\neigen vals\n",eigenvalue)
lambda1=eigenvalue[0]
print(eigenvalue[0])

lambda2=eigenvalue[1]
lambda3=eigenvalue[2]


# In[15]:


#D = np.array([lambda1,0, 0], [0, lambda2,0],[0,0,lambda3])
#in this section we are constructing the diagonal matrix accomodating the eigen values

a=lambda1
b=0
c=0
d=0
e=lambda2
f=0
g=0
h=0
i=lambda3
print("the diagonal matrix D is\n")
D = np.matrix('%s %s %s; %s %s %s; %s %s %s' % (a,b,c,d,e,f,g,h,i)) 
print(D)
D.shape


# In[16]:


P=np.matrix('%s ;%s ;%s ' % (eigenvector.transpose()[0],eigenvector.transpose()[1],eigenvector.transpose()[2])) 
print(P)

#print(P[0])
P.shape


# In[17]:


#in this section we are finding the inverse of matrix P

print("The matrix P inverse is \n \n")
print(P*-1)
print("\n")
print((P*-1).shape)


# In[10]:



#res = [[0 for x in range(3)] for y in range(3)]  
  
# explicit for loops 
#for i in range(len(P)): 
 #   for j in range(len(D[0])): 
  #      for k in range(len(D)): 
  #
   #         # resulted matrix 
    #        res[i][j] += P[i][k] * D[k][j] 
  
# print (res) 


# In[ ]:





# In[ ]:





# In[22]:


#Here we are converting all the matrices to list datatype to ease the matrix multiplication

R=P.tolist()
print("P converted to list is")
print("\n")
print(R)



S=D.tolist()
print("D converted to list is")
print("\n")
print(S)
      
result = [[0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]] 
  
# iterating by row of A 
for i in range(len(R)): 
  
    # iterating by coloum by B  
    for j in range(len(S[0])): 
  
        # iterating by rows of B 
        for k in range(len(S)): 
            result[i][j] += R[i][k] * S[k][j] 
print("\n \n")
for r in result: 
    print(r)
    print("\n")
    


# In[24]:



#we are converting P-1 to list to ease matrix multiplication

T=(P**(-1)).tolist()
print("\n")
print(T)
print("\n")
print(P**-1)


# In[30]:


result2 = [[0, 0, 0], 
        [0, 0, 0], 
        [0, 0, 0]] 
  

for i in range(len(result)): 
  
      
    for j in range(len(T[0])): 
  
         
        for k in range(len(T)): 
            result2[i][j] += result[i][k] * T[k][j] 
print("\n \n")
for r in result2: 
    print(r) 
print("\n")    
print(A)
print("\n")

print(" PDP-1 is \n")
x=np.ceil(result2[0])
y=np.ceil(result2[1])
z=np.ceil(result2[2])
arr2=([x,y,z])
print("\n")  
print(arr2)
p=np.ceil(A[0])
q=np.ceil(A[1])
r=np.ceil(A[2])
arr3=([p,q,r])
#print("A is\n ")
#print(arr3)


# In[16]:


print("after the first generation the probability of AA Aa and aa are as follows")


# In[34]:


#just verifying if the previous result is correct


arr_result = np.matmul(R,S)
my_matrix=arr_result

print(f'Matrix Product  is:\n{arr_result}')


# In[ ]:





# In[47]:


n=input("enetr the value of n")
n1=int(n)
print(n1)
my_matrix = np.matrix([[1, 0.5, 0], 
                       [0, 0.5, 1], 
                       [0, 0, 0]])
print(my_matrix.flatten())

for i in range(n1-1):


    arr_result = np.matmul(my_matrix, my_matrix)
    my_matrix=arr_result

print(f'Matrix Product  is:\n{arr_result}')


# In[44]:


#x=X.flatten()
#arr_result= arr_result.flatten()
#print(x)
#print(arr_result)
#final_ans=np.matmul(arr_result,x)
#print(final_ans)

#print(f'Matrix Product  is:\n{final_ans}')


# In[45]:


x.shape


# In[ ]:





# In[ ]:



#wE notice that after first generation the probabilities of Genotypes AA and Aa are 0.495 each which
#are approximately equal to 0.5(our theoretical result)


# In[48]:


A=np.array(A)
x=np.array(X)
my_matrix=np.array(my_matrix)
final_ans=np.matmul(my_matrix,X)
print(final_ans)
final_ans.shape


# In[ ]:




