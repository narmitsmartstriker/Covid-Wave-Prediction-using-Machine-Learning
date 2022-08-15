#Narmit Kumar


import pandas as pd 
import numpy as np
import math
import matplotlib.pyplot as plt
from math import sqrt

import warnings 
warnings.filterwarnings('ignore')

#Q2
print("Question 2 :")
#a Generating AR model 
from statsmodels.tsa.ar_model import AutoReg as AR

a = pd.read_csv('daily_covid_cases.csv')

#Train test split
series = pd.read_csv('daily_covid_cases.csv', parse_dates = ['Date'],index_col=['Date'],sep=',')
test_size = 0.35   #35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X) - tst_sz], X[len(X) - tst_sz:]

#Training AR model 
Window = 5  #The lag =5
model = AR(train, lags=Window)
model_fit = model.fit()  #fit/train the model
coef = model_fit.params  #Get the coefficients of AR model
print("The coefficient values are:")
print(coef.round(3))

#b part , predicting values
print("\n","B part")
print(" i and ii part -> Plots")
history = train[len(train) - Window:]
history = [history[i] for i in range (len(history))]
predictions = list()   # List to hold the predictions, 1 step at a time
for t in range(len(test)):
   length = len(history)
   lag = [history[i] for i in range(length-Window,length)]
   yhat = coef[0]   #Initialize to w0
   for d in range(Window):
       yhat += coef[d+1] * lag[Window-d-1]  #Add other values
   predictions.append(yhat)   #Append predicitions to compute RMSE
   obs = test[t]       
   history.append(obs)  #Append actual test value to history, to be used in next step      

#i Scatter plot   
plt.scatter(test,predictions, s=10)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Scatter plot between actual values and predicted values")
plt.show()

#ii Line plot
plt.plot(test,predictions, c="red")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Line plot between actual values and predicted values")
plt.show()


#iii RMSE(%) and MAPE between predicted and actual values.
s = 0
for i in range(len(test)):
    s = s+(predictions[i] - test[i])**2
avg = sum(test)/len(test)
rmspe = (sqrt(s/len(test))/avg)*100   #RMSE(%) value calculation

t = 0
for i in range(len(test)):
    t = t + abs(predictions[i] - test[i])/test[i]
mape = (t/len(test)) * 100        #MAPE value calculation
print("iii Part :")
print("The RMSE(%) value is {}".format(rmspe.round(3)))
print("The MAPE value is {}".format(mape.round(3)))


#Q3
print("\n")
print("Question 3")
n = [1,5,10,15,20,25]    #lag values for our model
RMSPE_values = []
MAPE_values = []
for z in n:
 Window = z
 model = AR(train, lags=Window)
 model_fit = model.fit()  #fit/train the model
 coef = model_fit.params  #Get the coefficients of AR model
 history = train[len(train) - Window:]
 history = [history[i] for i in range (len(history))]
 predictions = list()   # List to hold the predictions, 1 step at a time
 for t in range(len(test)):
   length = len(history)
   lag = [history[i] for i in range(length-Window,length)]
   yhat = coef[0]   #Initialize to w0
   for d in range(Window):
       yhat += coef[d+1] * lag[Window-d-1]  
   predictions.append(yhat)   
   obs = test[t]       
   history.append(obs)
 u = 0
 for h in range(len(test)):
     u = u + (predictions[h] - test[h])**2
 avg = sum(test)/len(test)
 rmspe = (sqrt(u/len(test))/avg)*100

 v = 0
 for h in range(len(test)):
     v = v + abs(predictions[h] - test[h])/test[h]
 mape = (v/(len(test))) * 100   
 RMSPE_values.append(rmspe)
 MAPE_values.append(mape)

print("RMSE(%) values are :")
for i in range(6):
   print("lag = {} &".format(n[i]),"Error = {}".format(RMSPE_values[i].round(3)))
print("\n")
print("MAPE values are :")
for i in range(6):
    print("lag = {} &".format(n[i]),"Error = {}".format(MAPE_values[i].round(3)))
f2 = []
f3 = []   
for i in range(6):
   f2.append(np.int(RMSPE_values[i]))
for j in range(6):   
   f3.append(np.int(MAPE_values[j])) 
print("\n")
print("Bar plots")
print("\n")
plt.bar(n,f2, width = 1.5 , color = "brown")
plt.title("RMSE(%) Values vs Lagged Values")
plt.show()   
plt.bar(n,f3, width = 1.5, color = "green")
plt.title("MAPE Values vs Lagged Values")
plt.show()
   
#Q4
print("Question_4")
df=pd.read_csv("daily_covid_cases.csv")
train_q4=df.iloc[:int(len(df)*0.65)]
train_q4=train_q4['new_cases']
i=0
corr = 1
# abs(AutoCorrelation) > 2/sqrt(T)
while corr > 2/(len(train_q4))**0.5:
    i += 1
    t_new = train_q4.shift(i)
    corr = train_q4.corr(t_new)
print("The optimal lag value is {}.".format(i - 1))

n1 = i - 1    #lag values for our model
RMSPE_value = 0
MAPE_value = 0

Window = n1
model = AR(train, lags=Window)
model_fit = model.fit()  #fit/train the model
coef = model_fit.params  #Get the coefficients of AR model
history = train[len(train) - Window:]
history = [history[i] for i in range (len(history))]
predictions = list()   # List to hold the predictions, 1 step at a time
for t in range(len(test)):
   length = len(history)
   lag = [history[i] for i in range(length-Window,length)]
   yhat = coef[0]   #Initialize to w0
   for d in range(Window):
       yhat += coef[d+1] * lag[Window-d-1]  
   predictions.append(yhat)   
   obs = test[t]       
   history.append(obs)
u = 0
for h in range(len(test)):
     u = u + (predictions[h] - test[h])**2
avg = sum(test)/len(test)
rmspe = (sqrt(u/len(test))/avg)*100

v = 0
for h in range(len(test)):
     v = v + abs(predictions[h] - test[h])/test[h]
mape = (v/(len(test))) * 100   
RMSPE_value += rmspe
MAPE_value += mape


print("RMSE(%) value is {}".format(RMSPE_value.round(3)))
print("MAPE value is {}".format(MAPE_value.round(3)))
    
    
    
    
