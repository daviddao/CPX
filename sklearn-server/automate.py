import pandas as pd 

## Models ##
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import grid_search

import numpy
from numpy import random
import time
from sklearn.datasets import load_iris

random.seed(1234)

class Automate():

	def __init__(self, models):
		# self.models = [RandomForestClassifier(n_estimators=100), OneVsRestClassifier(AdaBoostClassifier()), SVC(kernel="linear", class_weight="balanced", probability=True), SVC(class_weight="balanced", probability=True), LogisticRegression(), LDA(), KNeighborsClassifier()]
		# self.models = [SVC(kernel="linear", class_weight="balanced", probability=True), SVC(class_weight="balanced", probability=True)]
		
		# HT29
		# cv-accuracy for RandomForestClassifier: 0.857320132246 +- 0.0228867014846 and 75.2460298538 seconds
		# cv-accuracy for OneVsRestClassifier: 0.873197305821 +- 0.0231266134421 and 1985.48441601 seconds
		# cv-accuracy for LogisticRegression: 0.815473151385 +- 0.0205993400695 and 964.611626863 seconds
		# cv-accuracy for LinearDiscriminantAnalysis: 0.866086822815 +- 0.0269264556907 and 7.90767884254 seconds
		# cv-accuracy for KNeighborsClassifier: 0.471533159454 +- 0.0460108320998 and 2.27417302132 seconds
		# cv-accuracy for SVC: 0.785182492197 +- 0.0359244747131 and 877.674947023 seconds
		# cv-accuracy for SVC: 0.223047060237 +- 0.00556992351954 and 755.076722145 seconds

		self.models = models
		# cv-accuracy for GridSearchCV LinearSVC: 0.742999507874 +- 0.085105834868
		# cv-accuracy for GridSearchCV SVC: 0.746527436024 +- 0.0786473641483 and 3295.17767906 seconds
	def split_dataframe(self, df):
		tmp_df = df.copy() # make a copy to not alter original data
		#tmp_df.pop("ImageNumber") # throw away the indices
		#tmp_df.pop("ObjectNumber")
		y = tmp_df.pop("Class").values # default key
		y = map(int,y) # int conversion
		self.X = tmp_df.values 

		# Encode labels into numerical values
		le = preprocessing.LabelEncoder()
		le.fit(y)
		self.y = le.transform(y)

		return self.X, self.y 

	# Return name
	def name(self,model):
		return model.__class__.__name__

	def binarize(self):
		labels = list(set(self.y))
		self.y = label_binarize(self.y, classes=labels)[:,0] # some test data

	def split_test_train(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33)

	# Cheating to test hypothesis
	def gridsearchSVM(self):


		cv = KFold(len(self.y), n_folds=5, shuffle=True)

		estimator = SVC(kernel="linear", probability=True)
		# linear_clf = grid_search.GridSearchCV(estimator, param_grid={'C': [2**-3, 2**2]}, n_jobs=5)
		linear_clf = grid_search.GridSearchCV(SVC(kernel="linear", probability=True),param_grid={'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]},cv=5,n_jobs=5)
		rbf_clf = grid_search.GridSearchCV(SVC(probability=True),param_grid={'gamma': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1], 'C': [2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]},cv=cv,n_jobs=1)

		linear_clf.fit(self.X, self.y)
		rbf_clf.fit(self.X, self.y)

		# print linear_clf.grid_scores_
		l_params = linear_clf.best_params_
		# print l_params
		# print rbf_clf.grid_scores_
		r_params = rbf_clf.best_params_
		# print r_params

		self.models = [linear_clf, rbf_clf]		


	'''
	Train the models
	'''
	def train_models(self):

		json_training_time = []

		for model in self.models:
			start = time.time()
			model.fit(self.X_train, self.y_train) # Fit them to the data
			runtime = time.time() - start

			d = {}
			d["name"] = self.name(model)
			d["time"] = "{0:.5f}".format(runtime)
			json_training_time.append(d)

		return json_training_time 

	def get_cross_validation(self,folds):


		n = len(self.y)
		cv = KFold(n, n_folds=folds, shuffle=True)


		## TEST START
		n_cv = []
		for train,test in cv:
			n_cv.append(len(train))
		
		n_cv = min(n_cv)
		#cv = StratifiedKFold(y=self.y, n_folds=folds)
		json_cv_time = []

		grid_cv = KFold(n_cv, n_folds=5, shuffle=True)
		linear_clf = grid_search.GridSearchCV(SVC(kernel="linear", probability=True),param_grid={'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]},cv=grid_cv,n_jobs=-1)
		rbf_clf = grid_search.GridSearchCV(SVC(probability=True),param_grid={'gamma': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1], 'C': [2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]},cv=grid_cv,n_jobs=-1)

		self.models = [linear_clf, rbf_clf]
		## TEST END

		# Get accuracy
		def scorer(model, X, y):
			return model.score(X, y)

		for model in self.models:

			model_name = self.name(model)
			print "Now getting CV scores for", model_name
			start = time.time()

			scores = cross_val_score(model, self.X, self.y, cv=cv, scoring=scorer)

			runtime = time.time() - start
			print('cv-accuracy for {}: {} +- {} and {} seconds'.format(model_name, scores.mean(), scores.std(), runtime))
			d = {}
			d["name"] = self.name(model)
			d["time"] = "{0:.5f}".format(runtime)
			d["score"] = "{0:.3f}".format(scores.mean())
			d["std"] = "+-{0:.5f}".format(scores.std())
			json_cv_time.append(d)

		return json_cv_time 


	def get_accuracy(self):

		json_accuracy = []
		for model in self.models:
			model_name = self.name(model)
			score = model.score(self.X_test, self.y_test)
			# print('accuracy for {}: {}'.format(model_name, score))
			d = {}
			d["name"] = model_name
			d["score"] = "{0:.2f}".format(score) # limits to two digits
			json_accuracy.append(d) # append the dictionary

		return json_accuracy

	def get_precision_recall(self):

		# wrapper method for calculating the precision recall
		def get_precision_recall_curve(model):
			probas = model.predict_proba(self.X_test)
			precision, recall, thresholds = precision_recall_curve(self.y_test,
			                                                       probas[:, 1])
			#precision += [1]
			precision = sorted(precision)
			#recall += [0]
			recall = sorted(recall)[::-1]
			area = auc(recall, precision)
			print('Area under curve for {}: {:.3f}'.format(self.name(model), area))

			d = {}
			d["precision"] = []
			for el in precision:
				el_dict = {}
				el_dict["value"] = el
				d["precision"].append(el_dict)
			d["recall"] = []
			for el in recall:
				el_dict = {}
				el_dict["value"] = el
				d["recall"].append(el_dict)
			d["thresholds"] = []
			for el in thresholds:
				el_dict = {}
				el_dict["value"] = el
				d["thresholds"].append(el_dict)
			d["name"] = self.name(model) + ' (area = {:.3f})'.format(area)

			return d

		json_precision_recall = []
		for model in self.models:
			d = get_precision_recall_curve(model) # get the result as dictionary
			json_precision_recall.append(d)

		return json_precision_recall

	def get_roc(self):
		def get_roc_curve(model):
			probas = model.predict_proba(self.X_test)
			fpr, tpr, thresholds = roc_curve(self.y_test,
			                                 probas[:, 0],
			                                 pos_label=model.classes_[0])
			area = auc(fpr, tpr)

			d = {}
			d["tpr"] = []
			for el in tpr:
				el_dict = {}
				el_dict["value"] = el
				d["tpr"].append(el_dict)
			d["fpr"] = []
			for el in fpr:
				el_dict = {}
				el_dict["value"] = el
				d["fpr"].append(el_dict)
			d["thresholds"] = []
			for el in thresholds:
				el_dict = {}
				el_dict["value"] = el
				d["thresholds"].append(el_dict)
			d["name"] = self.name(model) + ' (area = {:.3f})'.format(area)

			return d

		json_roc = []
		for model in self.models:
			d = get_roc_curve(model) # get the result as dictionary
			json_roc.append(d)

		return json_roc





