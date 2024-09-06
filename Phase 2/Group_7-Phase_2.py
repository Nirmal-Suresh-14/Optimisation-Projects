#!/usr/bin/env python
# coding: utf-8

# # ME 609 : Optimisational Methods in Engineering
# ##     Project - Phase 2
# ## MultiVariable : Conjugate Direction Method
# ## Unidirectional Search : Bounding Phase, Newton Raphson
# ##   
# ###   Nirmal S.   [234103107]
# ###   Rohit Kumar Saragadam [234103109]

# In[1216]:


import numpy as np
import pandas as pd

import math


# ### Function Value

# In[1217]:


def function(df_x, df_s, alpha, N):
    
    global f_count     # Keeping Track of number of times the function is read
    f_count += 1
    
#     ########## Sum Squares Function ##########
#     value = 0
#     for i in range(1, N+1):
#         value += i*(df_x[i] + alpha*df_s[i])**2
    
#    ########## Rosenbrock Function ##########
#     value = 0
#     for i in range(1, N):
#         value += 100*(df_x[i+1] + alpha*df_s[i+1] - (df_x[i] + alpha*df_s[i])**2)**2 + (df_x[i] + alpha*df_s[i] - 1)**2
    
#     ########## Dixon Price Function ##########
#     value = (df_x[1] + alpha*df_s[1] - 1)**2
#     for i in range(2, N+1):
#         value += i*(2*(df_x[i] + alpha*df_s[i])**2 - df_x[i-1] + alpha*df_s[i-1])**2

#     ########## Trid Function ##########
#     value = 0
#     for i in range(1, N+1):
#         value += (df_x[i] + alpha*df_s[i] - 1)**2
#     for i in range(2, N+1):
#         value -= (df_x[i] + alpha*df_s[i])*((df_x[i-1] + alpha*df_s[i-1]))

    ########## Zakharov Function ##########
    value = 0
    for i in range(1, N+1):
        value += (df_x[i] + alpha*df_s[i])**2
    value_2 = 0
    for i in range(1, N+1):
        value_2 += 0.5*i*(df_x[i] + alpha*df_s[i])
    value += value_2**2
    value_2 = 0
    for i in range(1, N+1):
        value_2 += 0.5*i*(df_x[i] + alpha*df_s[i])
    value += value_2**4
    
    
    return value


# ### Unidirectional Search Algorithm Function

# In[1218]:


def uniDirectionalSearch(a, b, df_x, df_s):
    
    ########### INPUTS TO BOUNDING PHASE AND NEWTON RAPHSON ##########
    alpha = 0.
    delta = 1e-3
    
    h = 1e-4
    epsilon = 1e-5
    
    ########### BOUNDING PHASE METHOD ##########
    
    #### STEP 1 ####
    k = 0
    
    #### STEP 2 ####
    f_x = function(df_x, df_s, alpha, N)
    f_x_plus_delta  = function(df_x, df_s, alpha+delta, N)
    f_x_minus_delta = function(df_x, df_s, alpha-delta, N)
    if (f_x_minus_delta < f_x and f_x < f_x_plus_delta):
        delta = -delta
    alpha_prev = alpha

    while True:
        
        #### STEP 3 ####
        alpha_new = alpha + (2**k)*delta
        f_x_new = function(df_x, df_s, alpha_new, N)
        
        # if new guess goes beyond 'b'
        if alpha_new > b:    
            alpha_new = b
            f_x_new = function(df_x, df_s, alpha_new, N)
            if f_x_new >= f_x:
                a1 = alpha_prev 
                b1 = alpha_new
                break
            else:
                a1 = alpha 
                b1 = alpha_new
                break
                
        # if new guess goes below 'a'
        if alpha_new < a:    
            alpha_new = a
            f_x_new = function(df_x, df_s, alpha_new, N)
            if f_x_new >= f_x:
                b1 = alpha_prev 
                a1 = alpha_new
                break
            else:
                b1 = alpha 
                a1 = alpha_new
                break
        
        #### STEP 4 ####
        if f_x_new >= f_x:
            if delta < 0:
                b1 = alpha_prev 
                a1 = alpha_new
            else:
                a1 = alpha_prev 
                b1 = alpha_new
            break

        k += 1
        alpha_prev = alpha
        alpha = alpha_new
        f_x = f_x_new
    
    
    
    ########### NEWTON RAPHSON METHOD ##########
    
    #### STEP 1 ####
    k = 1
    alpha = b1
    f_d1 = ((function(df_x, df_s, alpha + h, N) 
             - function(df_x, df_s, alpha - h, N))
             /(2*h))
    
    while True:
        #### STEP 2 ####
        f_d2 = ((function(df_x, df_s, alpha + h, N)
                 - 2*function(df_x, df_s, alpha, N)
                 + function(df_x, df_s, alpha - h, N))
                /h**2)
        
        #### STEP 3 ####
        alpha_new = alpha - f_d1/f_d2
        f_new_d1 = ((function(df_x, df_s, alpha_new + h, N) 
                     - function(df_x, df_s, alpha_new - h, N))
                     /(2*h))
        
        #### STEP 4 ####
        
        if abs(f_new_d1) > epsilon:
            alpha = alpha_new
            f_d1 = f_new_d1
            continue
        
        else:
            return df_x + alpha_new*df_s, [alpha_new]


# ### Finding Bounds of Alpha

# In[1219]:


def findBound(a0, b0, df_x, df_s):
    min_max = pd.DataFrame(np.zeros((N,2)), index=range(1,N+1), columns=range(1,3))
    min_max[1] = -1000
    min_max[2] = 1000
        
    for bound_i in range(1, N+1):
        if abs(df_s[bound_i]) < 1e-2:
            continue
        else:
            min_max[1][bound_i] = (a0 - df_x[bound_i])/df_s[bound_i]
            min_max[2][bound_i] = (b0 - df_x[bound_i])/df_s[bound_i]
            
    return(max(min_max[1]), min(min_max[2]))


# ### Reading the initial point and the boundary from the input file

# In[1220]:


df_x = pd.DataFrame()     # To store values of each point

with open("Input_File.txt") as f:
    
    contents = f.read()
    
    x_start = contents.find("(") + 1
    x_end   = contents.find(")")
    
    bound_start = contents.find("[") + 1
    bound_end   = contents.find("]")
    
    # Storing initial point in dataframe and resetting index
    df_x[0] = np.array(contents[x_start:x_end].split(', '), dtype='float')
    df_x.index += 1
    
    # Reading the boundary of every variable
    bound = np.array(contents[bound_start:bound_end].split(', '), dtype='float')
    
    a0 = bound[0]
    b0 = bound[1]
    
    f.close()
    
# Extracting the number of variables in the input file
N = len(df_x)

# Creating the initial set of conjugate directions (Identity matrix)
df_s = pd.DataFrame(np.identity(N), columns=range(1,N+1), index=range(1, N+1))
    
print('The Initial Guess is \n', df_x[0].values)
print('\nThe bounds are [%.3f, %.3f]' %(a0, b0))
print('\nNumber of Variables : ', N)


# ## Powell's Conjugate Method

# In[1221]:


# Calculation of 'n'th point
n = 1

# Counting Number of function Evaluation:
f_count = 0

# Setting Value of Termination Checks
epsilon = 1e-3    # magnitude of 'd'
theta = 5    # angle in degrees

# DataFrame to set store alpha values of each direction
df_alpha = pd.DataFrame()

while True:
    
    #### STEP 2 ####
    # Running unidirectional searches for s1, s2, ..., sN
    for i in range(1, N+1):
        a1, b1 = findBound(a0, b0, df_x[n-1], df_s[i])
        df_x[n], df_alpha[n] = uniDirectionalSearch(a1, b1, df_x[n-1], df_s[i])
        n += 1
    #Running one more unidirectional search along s1
    a1, b1 = findBound(a0, b0, df_x[n-1], df_s[1])
    df_x[n], df_alpha[n] = uniDirectionalSearch(a1, b1, df_x[n-1], df_s[1])
     
    #### STEP 3 ####
    # Finding New conjugate Direction and it's magnitude
    d_new = df_x[n] - df_x[n-N]
    
    # Magnitude
    mod_d = 0
    for i in range(1, N+1):
        mod_d += d_new[i]**2
    mod_d = math.sqrt(mod_d)
    
    # Linear Dependency
    cos_sum = 0
    for i in range(1, N+1):
        cos_sum += (d_new/mod_d)[i] * df_s[1][i]
    cos_sum = 180/math.pi*math.acos(cos_sum)
    
    n += 1
    
    #### STEP 4 ####
    # Checking for Termination    
    if (mod_d > epsilon and cos_sum > theta):
        # Updating the search directions
        df_s.columns += 1
        df_s[1] = d_new/mod_d
        no_col = len(df_s.columns)
        df_s = df_s[range(1, no_col+1)]
    else:
        break


# ## Results:

# In[1222]:


print('***** RESULT *****\n')

print('The Optimum point at:')
for i in range(1, N+1):
    print('x'+str(i)+': %.3f' %df_x[df_x.columns.max()][i])
    
print('\nThe Mimumum value is:')
f_count_set = f_count
print('%.3f' %function(df_x[df_x.columns.max()], df_s[1], 0, N))

print('\nNumber of function Evaluations:')
print(f_count_set)


# In[1223]:


# print(df_x.round(3))
# print(df_alpha.round(3))
# print(df_s.round(3))


# ## Record Values

# In[1224]:


# # Create the DataFrame to record Data (run only once for each question)

# df_record = pd.DataFrame(columns=['Trial', 'Initial Point (x0)', 'Final Point (x*)', 'Function Value f(x*)', 'No. of Fn. Eval'])
# record_i = 1


# In[1225]:


# Record Data

# Initial Value
x0 = '('
for n in range(1, N+1):
    x0 += str('%.3f' %df_x[0][n]) + ', '
x0 = x0[:-2]
x0 += ')'

# Final Value
xf = '('
for n in range(1, N+1):
    xf += str('%.3f' %df_x[df_x.columns.max()][n]) + ', '
xf = xf[:-2]
xf += ')'


df_record.loc[record_i, 'Trial'] = record_i
df_record.loc[record_i, 'Initial Point (x0)'] = x0
df_record.loc[record_i, 'Final Point (x*)'] = xf
df_record.loc[record_i, 'Function Value f(x*)'] = '%.3f' %function(df_x[df_x.columns.max()], df_s[1], 0, N)
df_record.loc[record_i, 'No. of Fn. Eval'] = f_count_set

record_i += 1


# In[1226]:


function(df_x[0], df_s[df_s.columns.max()], 0, N)


# In[1227]:


df_record


# In[1230]:


# # Download Data of df_record

# df_record.to_csv('Outputs/Record_5.csv', index=False)


# In[1229]:


## Change in function value over successive points

df_fn_val = pd.DataFrame(columns=['No.', 'Point (x)', 'Fn_val f(x)'])

for i in df_x.columns:
    x = '('
    for n in range(1, N+1):
        x += str('%.3f' %df_x[i][n]) + ', '
    x = x[:-2]
    x += ')'
    
    df_fn_val.loc[i, 'No.'] = i
    df_fn_val.loc[i, 'Point (x)'] = x
    df_fn_val.loc[i, 'Fn_val f(x)'] = round(function(df_x[i], df_s[1], 0, N), 4)
    
df_fn_val
# df_fn_val.to_csv('Outputs/fn_val_5.csv', index=False)


# In[ ]:




