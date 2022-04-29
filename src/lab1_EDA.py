# Exploratory Data Analysis for the Housing dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('Portland_housing/ex3x.dat', header=None, sep='\s+', names=['LA', 'NB'])
df2 = pd.read_csv('Portland_housing/ex3y.dat', header=None, sep='\s+', names=['VAL'])

df = pd.concat([df1, df2], axis=1)

# print some values
df.head()

sns.set(style="ticks", color_codes=True)
cols1 = ['LA', 'VAL']
cols2 = df.columns
sns.pairplot(df[cols2])
sns.jointplot(x="LA", y="VAL", data=df)

# Compute the correlation coefficient
import numpy as np
plt.figure()
cm = np.corrcoef(df[cols2].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap( cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols2, xticklabels=cols2)
plt.show()

sns.reset_orig()

# plot some LA to MEDEV pairs
plt.figure()
plt.scatter(df['LA'], df['VAL'])
plt.xlabel('LA')
plt.ylabel('VAL')
plt.grid(True)
plt.show()

