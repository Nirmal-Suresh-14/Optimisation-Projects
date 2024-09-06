#!/usr/bin/env python
# coding: utf-8

# # ME 609 : Optimisational Methods in Engineering
# ##     Project - Phase 1 (Bounding Phase + Golden Section)
# 
# ###   Group 7
# ###   Nirmal S.   [234103107]
# ###   Rohit Kumar Saragadam [234103109]

# In[1]:


import numpy as np
import pandas as pd

import math

# import matplotlib.pyplot as plt


# In[2]:


def boundingPhase(a, b, x, delta, minmax):
    
    #### STEP 2 ####
    f_x = objectiveFunction(df_bound.loc[0, 'x_k'], minmax)
    f_x_plus_delta  = objectiveFunction(df_bound.loc[0, 'x_k']+delta, minmax)
    f_x_minus_delta = objectiveFunction(df_bound.loc[0, 'x_k']-delta, minmax)

    if (f_x_minus_delta < f_x and f_x < f_x_plus_delta):
        delta = -delta

    k = 0
    df_bound.loc[k, 'f(x_k)'] = f_x

    while True:
        
        #### STEP 3 ####
        df_bound.loc[k, 'x_(k+1)'] = df_bound.loc[k, 'x_k'] + (2**k)*delta
        df_bound.loc[k, 'f(x_(k+1))'] = objectiveFunction(df_bound.loc[k, 'x_(k+1)'], minmax)
        
        # if new guess goes beyond 'b'
        if df_bound.loc[k, 'x_(k+1)'] > b:    
            df_bound.loc[k, 'x_(k+1)'] = b
            df_bound.loc[k, 'f(x_(k+1))'] = objectiveFunction(df_bound.loc[k, 'x_(k+1)'], minmax)
            df_bound.loc[k, 'Continue/Terminate'] = "Terminate"
            
            if df_bound.loc[k, 'f(x_(k+1))'] >= df_bound.loc[k, 'f(x_k)']:
                return df_bound.loc[k-1, 'x_k'], df_bound.loc[k, 'x_(k+1)']
            
            else:
                return df_bound.loc[k, 'x_k'], df_bound.loc[k, 'x_(k+1)']
            
        # if new guess goes below 'a'
        if df_bound.loc[k, 'x_(k+1)'] < a:    
            df_bound.loc[k, 'x_(k+1)'] = a
            df_bound.loc[k, 'f(x_(k+1))'] = objectiveFunction(df_bound.loc[k, 'x_(k+1)'], minmax)
            df_bound.loc[k, 'Continue/Terminate'] = "Terminate"
            
            if df_bound.loc[k, 'f(x_(k+1))'] >= df_bound.loc[k, 'f(x_k)']:
                return df_bound.loc[k-1, 'x_k'], df_bound.loc[k, 'x_(k+1)']
            
            else:
                return df_bound.loc[k, 'x_k'], df_bound.loc[k, 'x_(k+1)']
        
        #### STEP 4 ####
        if df_bound.loc[k, 'f(x_(k+1))'] >= df_bound.loc[k, 'f(x_k)']:
            df_bound.loc[k, 'Continue/Terminate'] = "Terminate"
            break

        df_bound.loc[k, 'Continue/Terminate'] = "Continue"
        k += 1
        df_bound.loc[k,'k'] = k
        df_bound.loc[k, 'x_k'] = df_bound.loc[k-1, 'x_(k+1)']
        df_bound.loc[k, 'f(x_k)'] = df_bound.loc[k-1, 'f(x_(k+1))']

    return df_bound.loc[k-1, 'x_k'], df_bound.loc[k, 'x_(k+1)']
    
    


# In[3]:


def goldenSection(a1, b1, df_gold, epsilon, minmax):
           
        k = 0
        df_gold.loc[k, 'aw'] = 0
        df_gold.loc[k, 'bw'] = 1

        while True:
            df_gold.loc[k, 'lw'] = df_gold.loc[k, 'bw'] - df_gold.loc[k, 'aw']
            df_gold.loc[k, 'w1'] = df_gold.loc[k, 'aw'] + 0.618*df_gold.loc[k, 'lw']
            df_gold.loc[k, 'w2'] = df_gold.loc[k, 'bw'] - 0.618*df_gold.loc[k, 'lw']

            df_gold.loc[k, 'f(w1)'] = objectiveFunction(a1 + (b1 - a1)*df_gold.loc[k, 'w1'], minmax)
            df_gold.loc[k, 'f(w2)'] = objectiveFunction(a1 + (b1 - a1)*df_gold.loc[k, 'w2'], minmax)

            if df_gold.loc[k, 'lw']<=epsilon:
                df_gold.loc[k, 'Continue/Terminate'] = "Terminate"
                break

            df_gold.loc[k, 'Continue/Terminate'] = "Continue"

            if df_gold.loc[k, 'f(w1)'] < df_gold.loc[k, 'f(w2)']:
                df_gold.loc[k+1, 'aw'] = df_gold.loc[k, 'w2']
                df_gold.loc[k+1, 'bw'] = df_gold.loc[k, 'bw']

            else:
                df_gold.loc[k+1, 'aw'] = df_gold.loc[k, 'aw']
                df_gold.loc[k+1, 'bw'] = df_gold.loc[k, 'w1']

            k += 1

        if df_gold.loc[k, 'f(w1)'] < df_gold.loc[k, 'f(w2)']:
            return a1 + (b1 - a1)*df_gold.loc[k, 'w2'], a1 + (b1 - a1)*df_gold.loc[k, 'bw']

        else:
            return a1 + (b1 - a1)*df_gold.loc[k, 'aw'], a1 + (b1 - a1)*df_gold.loc[k, 'w1']


# In[4]:


def objectiveFunction(x, minmax):
    

#     f =  (2*x-5)**4 - (x**2-1)**3
#     f =  8 + x**3 -2*x -2*math.exp(x)
#     f =  4*x*math.sin(x)
#     f =  2*(x-3)**2 + math.exp(0.5*x**2)
    f =  x**2 - 10*math.exp(0.1*x)
#     f =  20*math.sin(x) - 15*x**2

    if minmax==1:
        return f
    return -f
    


# In[5]:


# Taking Values from the user for question number and boundary

while True:
    try:
        minmax = int(input("Select a values \n[1] Minimise Function\n[2] Maximise Function\n"))
        if(minmax<1 or minmax>2):     # Checking if the number is in the range of questions given
            print("Number out of bound, please try again")
            continue
        break
    except ValueError:
        print("Please enter an integer")
        
        
# print("\n", end='\n')
print("Enter the boundary values")

while True:
    try:
        a = float(input("Left boundary (a): "))
        break
    except ValueError:
        print("Please enter a number")
        
while True:
    try:
        b = float(input("Right boundary (b): "))
        if b<=a:     # Checking if b<=a
            print("Right boundary must be greater than the left, please try again")
            continue
        break
    except ValueError:
        print("Please enter a number")


# In[6]:


# Taking the initial guess and the delta value from user
df_bound = pd.DataFrame(columns=['k', 'x_k', 'x_(k+1)', 'f(x_k)', 'f(x_(k+1))', 'Continue/Terminate'], dtype='float')

while True:
    try:
        x_in = float(input("Enter the initial guess x(0): "))
        if x_in<a or x_in>b:     # Checking initial guess in [a,b]
            print("Not in the range of [a, b], please try again")
            continue
        break
    except ValueError:
        print("Please enter a number")
        
while True:
    try:
        delta = float(input("Enter the delta for Bounding Phase  (delta): "))
        break
    except ValueError:
        print("Please enter a number")

df_bound.loc[0,'k'] = 0
df_bound.loc[0,'x_k'] = x_in


# In[7]:


# Calling Bracketing Method:
a1, b1 = boundingPhase(a, b, df_bound, delta, minmax)

# print(round(a1,3), round(b1,3))

if minmax == 2:
    df_bound['f(x_k)'] = -df_bound['f(x_k)']
    df_bound['f(x_(k+1))'] = -df_bound['f(x_(k+1))']
    
df_bound


# In[8]:


# Calling Gradient Based Method:
df_gold  = pd.DataFrame(columns=['aw', 'bw', 'lw', 'w1', 'w2', 'f(w1)', 'f(w2)', 'Continue/Terminate'], dtype='float')

epsilon = 10**(-3)
a2, b2 = goldenSection(a1, b1, df_gold, epsilon, minmax)


# print(round(a2,5), round(b2,5))

if minmax == 2:
    df_gold['f(w1)'] = -df_gold['f(w1)']
    df_gold['f(w2)'] = -df_gold['f(w2)']
    
df_gold


# In[9]:


# Final Result:

if minmax==1: text_min_max = "Minimum" 
else: text_min_max =  "Maximum"

print("******* Result *******\n")    

print("From the Bounding Phase Method:")
print("Number of iterations: %d" %(len(df_bound)))
print("a = %.3f,  b = %.3f,  x(0) = %.3f,  delta = %0.3f" %(a, b, df_bound.loc[0,'x_k'], delta))
print("The region selected from Bounding Phase Method: (%0.4f, %0.4f)" %(a1, b1))


print("\n\nFrom the Golden Section Search Method:")
print("Number of iterations: %d" %(len(df_gold)))
print("a = %.4f,  b = %.4f,  epsilon = %.5f" %(a1, b1, epsilon))
print("The region selected from Bounding Phase Method: (%0.5f, %0.5f)" %(a2, b2))

