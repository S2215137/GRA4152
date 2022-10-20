## Testing
# 4.2. testerLinear.py

# Dependencies
import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


from linearmodels import *

real_estate = csvDataSet("real_estate.csv", scaled=True)

## Linear Regression Model 1
reg1 = LinearRegression()
reg1.linearModel(real_estate, "full", "y ~ b0 + b1*x1")
reg1.optimize()
reg1.summary()

# Creating an instance of Linear Regression 1 model on data.
reg1_model_plot = diagnosticPlot(reg1)
# Plotting the Linear Regression with y-values vs mu.

reg1_model_plot.plot(real_estate.y, reg1.predict(real_estate.x))


## Linear Regression Model 2
reg2 = LinearRegression()
reg2.linearModel(real_estate, "full", "y ~ b0 + b1*x2 + b2*x3 + b3*x4")
reg2.optimize()
reg2.summary()
# Creating an instance of Linear Regression 1 model on data.
reg2_model_plot = diagnosticPlot(reg2)
# Plotting the Linear Regression with y-values vs mu.
reg2_model_plot.plot(real_estate.y, reg2.predict(real_estate.x))



## Linear Regression Model 3
reg3 = LinearRegression()
reg3.linearModel(real_estate, "full" ,"y ~ b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4 + b5*x5" )
reg3.optimize()
reg3.summary()

# Creating another instance of Linear Regression model 3 on the dataset.
reg3_model_plot = diagnosticPlot(reg3)

# Plotting the Linear Regression with y-values vs mu.
reg3_model_plot.plot(real_estate.y, reg3.predict(real_estate.x))


## Linear Regression Model 4
reg4 = LinearRegression()
reg4.linearModel(real_estate, "full" ,"y ~ b1*x1" )
reg4.optimize()
reg4.summary()

# Creating another instance of Linear Regression model 3 on the dataset.
reg4_model_plot = diagnosticPlot(reg4)

# Plotting the Linear Regression with y-values vs mu.
reg4_model_plot.plot(real_estate.y, reg4.predict(real_estate.x))