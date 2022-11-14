# All the required classes are in the additional 
# classes_for_last_exercise.py file, to display the code is working.

# Example terminal code that works:
# python last_exercise.py  -perc_training_set 70 -covar 0 1 -make_plot 

## Testing
# 4.1. testerLogistic.py
# Dependencies
import textwrap
import argparse
import random
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from linearmodels import *
import statsmodels.api as sm

## Adding code description and filtering for the sutiable inputs for the program.

parser = argparse.ArgumentParser(
    description=textwrap.dedent('''This program is a test for the logistic 
    regression model developed in linearmodels.py.'''
), 
    epilog=textwrap.dedent('''And that's how this tester file for logistic
    regression model works.
    '''))

# Seed value, which is by default 12345
parser.add_argument('-seed_value', '-seed_val', type=int, default=12345,
help=textwrap.dedent('''A random state from the random module, to split the training and
 test set. Default value is 12345'''))

# Percentage of the training set that should be included.
parser.add_argument('-perc_training_set', '--perc_training_set',type=int, 
help='The percentage of the split of training set.')

# Covariates can be specified chosen between 1, 2, and 3. If the user wants a constant, 0 should be included as well.
parser.add_argument('-covar', '--covariates', nargs='+', type=int, choices=[0, 1, 2, 3],
    help=textwrap.dedent('''Write the wanted covariates/independent variables in the formula in a list
    format input, should look like this for example: 1 3, means columns 1 and 3
    are the covariates that should be used in this case.
    0 in this case would be to add a constant to the regression model.'''))

# To specify the output will include a plot, include -make_plot
parser.add_argument('-make_plot', '--make_plot', action='store_true', 
    help='Specify this value if the user wants the output to be plotted. Specify this variable'+
    'to output the plot.')

args = parser.parse_args()

import statsmodels.api as sm
spector_data = sm.datasets.spector.load()
# if pandas dataset in statsmodels (some versions of statsmodels):
spector_y = spector_data.endog.values
spector_x = spector_data.exog.values
# if numpy arrays in statsmodels (some versions of statsmodels)
#spector_y = spector_data.endog
#spector_x = spector_data.exog

spector = DataSet(spector_x, spector_y)

# Splitting the dataset.
spector.train_test(testSize = abs(1-args.perc_training_set)/100, randomState=args.seed_value)

## Logistic Regression Model 1
# Creating an instance of the first regression model.
reg1 = LogisticRegression()
# Specifying dataobject, covariates and the training set to use in the
# model.
s = 'y ~ '
for count, covar in enumerate(args.covariates):
    covar = str(covar)
    if covar == '0':
        s += f"b{covar}"
    else:
        s += f"b{covar}*x{covar}"
    if count == len(args.covariates) -1:
        break
    s += ' + '

reg1.linearModel(spector, 'train', s)

# Fitting the beta parameter's.
reg1.optimize()
reg1.summary()


# Creating an instance of diagnosticPlot with Linear Regression
# as input.
reg1_model_plot = diagnosticPlot(reg1)

yTe = spector.y_te.reshape(-1,1)
# Predicting the Logistic Regression 1 model on the test set.
x_te = spector.x_te
y_pred = reg1.predict(x_te)
# Plotting the ROC curve on the test set.
if args.make_plot:
    reg1_model_plot.plot(yTe, y_pred)