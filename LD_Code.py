import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style

filename_r = 'data//winequality-red.csv'
filename_w = 'data//winequality-white.csv'

n = 4898 # number of white wines
d = 12
data = []
features = []
# let's just look at white wines for now
# with open(filename_r, newline='') as file:
# 	reader = csv.reader(file, delimiter = ';')
# 	for i, row in enumerate(reader):
# 		if i == 0:
# 			features = row
# 		else:
# 			data.append(row)

with open(filename_w, newline='') as file:
	reader = csv.reader(file, delimiter = ';')
	for i, row in enumerate(reader):
		if i == 0:
			features = row
		else:
			data.append(row)

# at this point, our data includes strings and numbers (still scored as strings)
# we need to convert everything to numeric values, then put in an np.array
# for example, change 'FALSE' -> 0, 'TRUE' -> 1, and use some sort of encoding for 'Visitor_Type'

# the last row of our data is QUALITY: let's try to predict whether a wine is above or below average
X = np.zeros((n,d-1))
Y = np.zeros(n) 
for i in range(n):
	if float(data[i][-1]) >= 6: Y[i] = 1 # good wines are calssified as 1
	else: Y[i] = 0 # bad wines are classified as 0
	for j in range(d-1):
		X[i,j] = float(data[i][j])

# TODO: should probably rescale/normalize each feature to be on the same scale

num_good = np.sum(Y) 
# Someone like me who doesn't know much about wine and thinks they all taste pretty good would classify each wine as a 1
# This is our 'null model', which would have accuracy ~ num_good/n

# Reduce data dimention using PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Plot results from PCA
print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
plt.figure()
colors = ['red', 'blue']
lw = 2
target_names = np.unique(Y)
for color, i, target_name in zip(colors, target_names, target_names):
        plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color = color, alpha=.5, lw=lw, label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of wines')

# separate into training and testing data
num_train = int(.8 * n)
num_test = n - num_train
X_train = X_r[0:num_train, :]
X_test = X_r[num_train:,:]
Y_train = Y[0:num_train]
Y_test = Y[num_train:]

# now we're ready to train models on the data X_train and Y_train
print('Features: ', features)
print('Data: ', X_train[0:5,:])
print('Labels: ', Y[0:100])
plt.show()
