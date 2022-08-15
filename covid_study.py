#Narmit Kumar


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("daily_covid_cases.csv")
A = a.copy()

print("Question 1")
#A Line plot between days and number of covid cases
fig, ax= plt.subplots()
x = range(1,613)
y = a['new_cases']
ax.plot(x,y)

month_starts = [3,63,124,185,246,307,369,428,489,550,611]
labels = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21',"OCt-21"]
ax.set_xticks(month_starts)
ax.set_xticklabels(labels, fontsize = 8)
plt.xlabel("Month-Year")
plt.ylabel("New confirmed cases")
plt.title("Lineplot--Q1a")
plt.show()

#B Lag sequence for 1 day and it's correlation
A['lag_1'] = A['new_cases'].shift(1)
Cor1 = A['lag_1'].corr(A['new_cases'])
print("Pearson correlation (autocorrelation) coefficient between the generated one-day lag time sequence and the given time sequence:",Cor1.round(3))

#C Scatter plot between lag_1 and original time sequence
plt.scatter(A['new_cases'],A['lag_1'], s=20, c="maroon")
plt.xlabel("Orginal time sequence")
plt.ylabel("One day lagged generated sequence")
plt.title("Scatter-plot Q1c")
plt.show()

#D Generating multiple time sequences with lag of 1,2,3, upto 6 days
x1 = [1,2,3,4,5,6]
corr_values = []
for i in range(1,7):
    A['lag_{}'.format(i)] = A['new_cases'].shift(i)
    n = A['lag_{}'.format(i)].corr(A['new_cases'])
    corr_values.append(n)
print("\n")
print("Pearson correlation coefficient between each of the generated time sequences and the given time sequence")
for i in range(6) : 
    print("Lag = {} & ".format(x1[i]), "Correlation = {} ".format(corr_values[i].round(3)))
#Plotting line plot between correlationcoefficients and lagged values   
plt.plot(x1,corr_values, c='gold')
plt.xlabel("Lagged Values")
plt.ylabel("Obtained Correlation Coefficients")
plt.title("Lineplot--Q1d")
plt.show()

#E 
#Plotting a correlogram or Auto Correlation Function using function ‘plot_acf’
from statsmodels.graphics.tsaplots import plot_acf
ax1 = plot_acf(A['new_cases'], lags = 6)
ax2 = plot_acf(A['new_cases'],lags =6)
plt.ylim(0.960,1)


