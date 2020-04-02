import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.linear_model import LogisticRegression as LR # using sklearn instead
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from scipy.linalg import eigh
from sklearn import metrics
from tqdm import tqdm
import sklearn.metrics as met


def p_1(test, train_X, train_y, test_X, test_y):
	train_N = len(train_y)

	#5-fold cross validation
	C = [.1, 1, 10, 100, 1000]
	c_score_val = [0,0,0,0,0]
	c_score_train = [0,0,0,0,0]
	part_size = train_N // 5
	for i, c in enumerate(C):
		total_score = 0
		train_score = 0
		for j in range(5):
			a = j*part_size
			b = a + part_size-1
			x_dat = np.concatenate((train_X[0:a,:], train_X[b:,:]))
			y_dat = np.concatenate((train_y[0:a], train_y[b:]))
			M = LR(penalty='l2', solver='liblinear', C=c).fit(X=x_dat, y=y_dat)
			total_score += M.score(train_X[a:b+1,:], train_y[a:b+1])
			train_score += M.score(x_dat, y_dat)
		c_score_val[i] = total_score/5
		c_score_train[i] = train_score/5
	print("train error: ", c_score_train)
	print("validation error: ", c_score_val) 
	# Best C for cancer -> 10
	# Best C for sonar -> 1000

	best_C = 10
	model = LR(penalty='l2', solver='liblinear', C=best_C).fit(X=train_X, y=train_y)
	test_acc = model.score(test_X, test_y)
	print(test_acc)


def p_2(test, train_X, train_y, test_X, test_y, kernel='linear'):
	train_N = len(train_y)

	#5-fold cross validation
	C = [.1, 1, 10, 100, 1000]
	c_score_val = [0,0,0,0,0]
	c_score_train = [0,0,0,0,0]
	part_size = train_N // 5
	for i, c in enumerate(C):
		total_score = 0
		train_score = 0
		for j in range(5):
			a = j*part_size
			b = a + part_size-1
			x_dat = np.concatenate((train_X[0:a,:], train_X[b:,:]))
			y_dat = np.concatenate((train_y[0:a], train_y[b:]))
			M = SVC(C=c, kernel=kernel).fit(X=x_dat, y=y_dat)
			total_score += M.score(train_X[a:b+1,:], train_y[a:b+1])
			train_score += M.score(x_dat, y_dat)
		c_score_val[i] = np.round_(total_score/5, decimals=3)
		c_score_train[i] = np.round_(train_score/5, decimals=3)
	print("train accuracy: ", c_score_train)
	print("validation accuracy: ", c_score_val) 

	best_C = 100 # 
	print("Fitting: ")
	model = SVC(C=best_C, kernel=kernel, cache_size=7000).fit(X=train_X, y=train_y)
	print("Testing: ")
	test_acc = model.score(test_X, test_y)
	print(test_acc)


def p_3(test, train_X, train_y, test_X, test_y, method='norm'):
	new_train = np.zeros(train_X.shape)
	new_test = np.zeros(test_X.shape)
	train_N = len(train_y)

	# I saved the pre-processed data to save time
	# uncomment this block to do pre-processing 
	# for j in range(54):
	# 	col = train_X[:,j]
	# 	mean = float(np.mean(col))
	# 	std = float(np.std(col))
	# 	big = float(np.max(col))
	# 	small = float(np.min(col))
	# 	if std == 0: 
	# 		std = 1
	# 	if big==small:
	# 		big = 0.1
	# 	for i in range(522911):
	# 		if method == 'resc': #rescaling
	# 			new_train[i,j] = (train_X[i,j] - small)/(big - small)
	# 		elif method == 'stan': #standardization
	# 			new_train[i,j] = (train_X[i,j] - mean)/(big - small)
	# 		else: #normalization
	# 			new_train[i,j] = (train_X[i,j] - mean)/std
	# for j in range(54):
	# 	col = test_X[:,j]
	# 	mean = float(np.mean(col))
	# 	std = float(np.std(col))
	# 	big = float(max(col))
	# 	small = float(min(col))
	# 	# all entries in test data for column 29 are 0 -> this fixes it
	# 	if std == 0: 
	# 		std = 1
	# 	if big==small:
	# 		big = 0.1
	# 	for i in range(58101):
	# 		if method == 'resc': #rescaling
	# 			new_test[i,j] = (test_X[i,j] - small)/(big - small)
	# 		elif method == 'norm': #standardization
	# 			new_test[i,j] = (test_X[i,j] - mean)/(big - small)
	# 		else: #standardization
	# 			new_test[i,j] = (test_X[i,j] - mean)/std
	# np.save('stan_train.npy', new_train)
	# np.save('stan_test.npy', new_test)

	new_train = np.load(method+'_train.npy')
	new_test = np.load(method+'_test.npy')
	# print("Doing cross-validation")
	# C = [.1, 1, 10]
	# c_score_val = [0,0]
	# part_size = train_N // 5
	# for i, c in enumerate(C):
	# 	print("c=", c)
	# 	total_errors = 0
	# 	for j in range(5):
	# 		a = j*part_size
	# 		b = a + part_size-1
	# 		x_dat = np.concatenate((new_train[0:a,:], new_train[b:,:]))
	# 		y_dat = np.concatenate((train_y[0:a], train_y[b:]))
	# 		M = LinearSVR(C=c, dual=True).fit(X=x_dat, y=y_dat)
	# 		predictions = M.predict(train_X[a:b+1,:])
	# 		num_errors = 0.5 * sum(abs(np.sign(predictions) - train_y[a:b+1]))
	# 		total_errors += num_errors	
	# 	c_score_val[i] = total_errors
	# print("Number of errors in validation for each C: ", c_score_val) 

	best_C = 0.1 # 	
	print("Fitting: ")
	# this linear SVR is must faster because it uses regression to approximate (or something)
	model = LinearSVR(C=best_C, dual=True).fit(X=new_train, y=train_y)
	print("Testing: ")
	predictions = model.predict(new_test) 
	acc = met.accuracy_score(test_y, np.sign(predictions))
	print("Test accuracy: ", acc)
	f1 = met.f1_score(test_y, np.sign(predictions))
	print("F-1 Score: ", f1)
	auc = met.roc_auc_score(test_y, predictions)
	print("AUC Score: ", auc)

	fpr, tpr, thresh = met.roc_curve(test_y, predictions)
	plt.plot(fpr,tpr)
	plt.title("ROC Curve")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.show()



#######################################################
def main():
	# Parse command-line args
    parser = argparse.ArgumentParser(description="ml_hw_5")
    parser.add_argument("--prob_num", action="store", type=str, default="1",
                        required=True, help="Problem number for ML HW 5")
    parser.add_argument("--test", action="store", type=str, default="F",
    					required=False, help="T -> test time, and F -> train")
    parser.add_argument("--data", action="store", type=str, default="cancer",
    					required=False, help="\'cancer\' data or \'sonar\' data")
    parser.add_argument("--kernel", action="store", type=str, default="linear",
    					required=False, help="\'linear\' or \'poly\' or \'rbf\'")
    parser.add_argument("--method", action="store", type=str, default="resc",
    					required=False, help="\'resc\' or \'stan\' or \'norm\'")

    problem = parser.parse_args().prob_num
    test = ('T' == parser.parse_args().test)
    data = parser.parse_args().data
    kernel = parser.parse_args().kernel
    method = parser.parse_args().method

    # Load data
    if problem == '3':
    	# 581012 total points
    	data_file = open('homework-5-data\\covtype.data', 'r')
    	train_ind_file = open('homework-5-data\\covtype.train.index.txt', 'r')
    	test_ind_file = open('homework-5-data\\covtype.test.index.txt', 'r')
    	all_data = np.zeros((581012, 54))
    	all_labels = np.zeros(581012)
    	for count, line in enumerate(data_file):
    		items = line.split(',')
    		for i, item in enumerate(items):
    			if i == 54:
    				# we want to classify label '2' as positive, all else as negative
    				if int(item) == 2:
    					all_labels[count] = 1
    				else:
    					all_labels[count] = -1
    			else:
    				all_data[count, i] = float(item)
    	train_data = np.zeros((522911,54))
    	test_data = np.zeros((58101,54)) 
    	train_labels = np.zeros(522911)
    	test_labels = np.zeros(58101)
    	for count, line in enumerate(train_ind_file):
    		idx = int(line) - 1
    		train_data[count,:] = all_data[idx,:]
    		train_labels[count] = all_labels[idx]
    	for count, line in enumerate(test_ind_file):
    		idx = int(line) - 1
    		test_data[count,:] = all_data[idx,:]
    		test_labels[count] = all_labels[idx] 	

    elif data == 'cancer':
    	# 683 total points
    	data_file = open('homework-5-data\\breast-cancer_scale.txt', 'r')
    	train_ind_file = open('homework-5-data\\breast-cancer-scale-train-indices.txt', 'r')
    	test_ind_file = open('homework-5-data\\breast-cancer-scale-test-indices.txt', 'r')
    	all_data = np.zeros((683, 10))
    	all_labels = np.zeros(683)
    	for count, line in enumerate(data_file):
    		items = line.split(' ')
    		all_labels[count] = int(items[0]) - 3 # labels are 2s and 4s
    		items = items[1:11]
    		for i, item in enumerate(items):
    			text = item.split(':')
    			val = float(text[1])
    			all_data[count, i] = val
    	train_data = np.zeros((500,10))
    	test_data = np.zeros((183,10)) 
    	train_labels = np.zeros(500)
    	test_labels = np.zeros(183)
    	for count, line in enumerate(train_ind_file):
    		idx = int(line) - 1
    		train_data[count,:] = all_data[idx,:]
    		train_labels[count] = all_labels[idx]
    	for count, line in enumerate(test_ind_file):
    		idx = int(line) - 1
    		test_data[count,:] = all_data[idx,:]
    		test_labels[count] = all_labels[idx]

    elif data == 'sonar':
    	# 208 total points
    	data_file = open('homework-5-data\\sonar_scale.txt', 'r')
    	train_ind_file = open('homework-5-data\\sonar-scale-train-indices.txt', 'r')
    	test_ind_file = open('homework-5-data\\sonar-scale-test-indices.txt', 'r')
    	all_data = np.zeros((208, 60))
    	all_labels = np.zeros(208)
    	for count, line in enumerate(data_file):
    		items = line.split(' ')
    		all_labels[count] = int(items[0])
    		items = items[1:61]
    		for i, item in enumerate(items):
    			text = item.split(':')
    			if text[0] != "\n": # one row in the data was missing an item, screwed everything up
    				val = float(text[1])
    				all_data[count, i] = val
    	train_data = np.zeros((150,60))
    	test_data = np.zeros((58,60)) 
    	train_labels = np.zeros(150)
    	test_labels = np.zeros(58)
    	for count, line in enumerate(train_ind_file):
    		idx = int(line) - 1
    		train_data[count,:] = all_data[idx,:]
    		train_labels[count] = all_labels[idx]
    	for count, line in enumerate(test_ind_file):
    		idx = int(line) - 1
    		test_data[count,:] = all_data[idx,:]
    		test_labels[count] = all_labels[idx]

    if problem =='1':
    	p_1(test=test, train_X=train_data, train_y=train_labels, test_X=test_data,
    		test_y=test_labels)
    elif problem =='2':
    	p_2(test=test, train_X=train_data, train_y=train_labels, test_X=test_data,
    		test_y=test_labels, kernel=kernel)
    elif problem =='3':
    	p_3(test=test, train_X=train_data, train_y=train_labels, test_X=test_data,
    		test_y=test_labels, method=method)
    else: 
    	print("Enter a valid problem number (1,2,3).")

if __name__ == '__main__':
    main()