import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# wine type
w_type = 'white'

# Load training data
col_names = ['FXA', 'VLA', 'CTA', 'RSG', 'CHL', 'FSD', 'TSD', 'DN', 'PH', 'SPH', 'ALH', 'Q']
xcols = ['FXA', 'VLA', 'CTA', 'RSG', 'CHL', 'FSD', 'TSD', 'DN', 'PH', 'SPH', 'ALH']
df_tr = pd.read_csv('/home/thomas/ML&DM/Code/Labs/Lab 2/wine_quality/winequality-' + w_type + '_tr.csv', header=0, index_col=None, names=col_names)
df_tr.head()

# Compute normalizers
tr_mean = df_tr.mean()
tr_std = df_tr.std()

# ---------------------------------------------------------------
# Compute LR model
# ---------------------------------------------------------------

# ... insert your code here...

# ---------------------------------------------------------------
# Compute errors on training set:
# ---------------------------------------------------------------
# Mean Squared Error (MSE)
MSE_tr = 1.0/m_tr * np.sum(np.square(Y - Y_hat))

# Residual Standard Error (RSE)
RSE_tr = np.sqrt( 1.0/(m_tr - n -1) * np.sum(np.square(Y - Y_hat)  ))

print("Training error")
print("MSE: ", MSE_tr, "RMSE: ", RSE_tr)

# plot Y vs Y_hat
plt.figure()
plt.scatter(Y, Y_hat)
# NB: m_tr is the number of samples in the training set, n is the number of
# features used in the regression, not counting the intercept!

# ---------------------------------------------------------------
# Test
# ---------------------------------------------------------------
#load test data
df_te = pd.read_csv('/home/thomas/ML&DM/Datasets/Wine_quality/winequality-' + w_type + '_te.csv', header=0, index_col=None, names=col_names)
df_te.head()

# normalize data, if needed
# df_te = (df_te - tr_mean)/tr_std

# Select, or add features and create design matrices (X and Y)
# ...your code here...
# X_te = ...

# Test
Y_te_hat = np.dot(X_te,W)

# compute errors
MAD_te = 1.0/m_te * np.sum(np.abs(Y_te - Y_te_hat))
MSE_te = 1.0/m_te * np.sum(np.square(Y_te - Y_te_hat))
RSE_te = np.sqrt( 1.0/(m_te - n -1) * np.sum(np.square(Y_te - Y_te_hat)))

print("Test error")
print("MSE: ", MSE_te, "RMSE: ", RSE_te, "MAE: ", MAD_te)

# plot Y vs Y_hat
plt.figure()
plt.scatter(Y_te, Y_te_hat)