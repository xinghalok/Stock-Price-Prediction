import quandl as qd
import numpy as np
import datetime 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#getting data from Quandal as its index is datatime dtype
data=qd.get("EOD/MSFT",auth_token="sG1gRzNEgoDwzJbfm22k")
#taking only the Adj values as they are precomputed statistical data
data=data[["Adj_Open","Adj_Close","Adj_High","Adj_Low","Adj_Volume"]]

#our Assumptions of percent of change on data
#if the change is one pct
pct=int(len(data)*0.001)
#now we will create a new close on the pct change with shift of DF
data["newclose"]=data["Adj_Close"].shift(-pct)
#-ve shift is shifting th columuns upwards results in nan at bottom
#+ve shift is shifting th columuns upwards results in nan at bottom
#extracting all data except newclose as x
x=data.drop("newclose",1)
#scaling all float values to same digit precision
x=scale(x)
#taking all data except last pct data for train and test
x_a=x[:-pct]
#taking last pct values as we have nan in newclose for these x
x_p=x[-pct:]
#extract all newclose except nan as y
y=data.dropna()["newclose"]
#split the x and y as train and test data so that m and b can be
#calculated from train and can be tested with test data
x_r,x_t,y_r,y_t=train_test_split(x_a,y,train_size=0.7)

alg=LinearRegression(n_jobs=10)
alg.fit(x_r,y_r)
print(alg.score(x_t,y_t))
data["Adj_Close"].plot()
data["forecast"]=np.nan  # all the coloumns will filled with nan #
lastday=data.iloc[-1].name.timestamp()
od=86400
val=alg.predict(x_p)
for i in val:
    lastday=lastday+od
    nextday=datetime.datetime.fromtimestamp(lastday)
    data.loc[nextday]=[np.nan for k in range(6)]+[i]
data["forecast"].plot()
plt.show()

