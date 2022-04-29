import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

col_names = ['FXA', 'VLA', 'CTA', 'RSG', 'CHL', 'FSD', 'TSD', 'DN', 'PH', 'SPH', 'ALH', 'Q']
xcols = ['FXA', 'VLA', 'CTA', 'RSG', 'CHL', 'FSD', 'TSD', 'DN', 'PH', 'SPH', 'ALH']
df = pd.read_csv('/home/thomas/ML&DM/Code/Labs/Lab 2/wine_quality/winequality-white_tr.csv', header=0, index_col=None, names=col_names)

(m_tr,n) = df[xcols].shape

# df[xcols] = (df[xcols]-df[xcols].mean())/df[xcols].std()

df['I']=1.0
df.head()
df = df.dropna()


# plot correlation matrix
pcols = ['FXA', 'VLA', 'RSG', 'FSD', 'DN', 'PH', 'SPH', 'ALH','Q']
plt.figure()
cm = np.corrcoef(df[pcols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap( cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=pcols, xticklabels=pcols)
plt.show()

# plot scatterplots
sns.pairplot(df[pcols])

# Train model
xcols = ['FXA', 'VLA', 'RSG', 'SPH', 'ALH']
X = df[['I'] + xcols].as_matrix()
Y = df['Q'].as_matrix()
results = sm.OLS(Y, X).fit()
print results.summary()

# prediction
W = results.params
Y_hat = np.dot(X,W)

# Mean Squared Error (MSE)
MSE_tr = 1.0/m_tr * np.sum(np.square(Y - Y_hat))

# Residual Standard Error (RSE)
RSE_tr = np.sqrt( 1.0/(m_tr - n -1) * np.sum(np.square(Y - Y_hat)  ))

print MSE_tr, RSE_tr
plt.figure()
plt.scatter(Y,Y_hat)