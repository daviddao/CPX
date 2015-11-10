import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
import numpy
from numpy import random

random.seed(1234)

class Automate():

	def init(self):
		self.awesome = "Awesome!"

	def split_dataframe(self, df):
		tmp_df = df.copy() # make a copy to not alter original data
		tmp_df.pop("ImageNumber") # throw away the indices
		tmp_df.pop("ObjectNumber")
		self.y = tmp_df.pop("Class").values # default key
		self.X = tmp_df.values 

		return self.X, self.y 

	# Return name
	def name(self,model):
		return model.__class__.__name__

	# Split data into test and training set
	def split_train_test(self, train_percent=0.6):
		split_index = int(len(self.X) * train_percent)

		# Combine features and labels into a single list so we can shuffle together
		points = zip(self.X, self.y)
		random.shuffle(points)
		training_points = points[:split_index]
		test_points = points[split_index:]

		# Now that our points have been randomly ordered, separate the features from the labels again
		# Also put them in numpy arrays.
		self.X_train = numpy.array([X_point for (X_point, y_point) in training_points])
		self.y_train = numpy.array([y_point for (X_point, y_point) in training_points])

		self.X_test = numpy.array([X_point for (X_point, y_point) in test_points])
		self.y_test = numpy.array([y_point for (X_point, y_point) in test_points])

		return self.X_train, self.y_train, self.X_test, self.y_test


	def init_models(self):
		self.models = [RandomForestClassifier(), AdaBoostClassifier(), SVC(probability=True), GradientBoostingClassifier(), LogisticRegression(), LDA(), KNeighborsClassifier()]

		for model in self.models:
			model.fit(self.X_train, self.y_train) # Fit them to the data

		return self.models

	def get_accuracy(self):
		json_accuracy = []
		for model in self.models:
			model_name = self.name(model)
			score = model.score(self.X_test, self.y_test)
			print('accuracy for {}: {}'.format(model_name, score))
			d = {}
			d["name"] = model_name
			d["score"] = "{0:.2f}".format(score) # limits to two digits
			json_accuracy.append(d) # append the dictionary


		return json_accuracy



