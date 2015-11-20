#!flask/bin/python

from automate import Automate

from flask import Flask, jsonify, make_response, request
from flask_restful import Resource, Api, reqparse # Flask API
import pandas as pd 




app = Flask(__name__)
api = Api(app) # Generate an API

###############
#### Model ####
###############  

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

rdf = RandomForestClassifier(n_estimators=100)
ada = OneVsRestClassifier(AdaBoostClassifier())
lsvc = SVC(kernel="linear", C=4, probability=True)
rsvc = SVC(probability=True, C=32, gamma=0.125) # tuned HP for HELA
log = LogisticRegression()
lda = LDA()
knn = KNeighborsClassifier()

# estimators=[('lr', rdf), ('rf', log), ('lda', lda)]
estimators=[('l',rdf),('lda', lda),('svc',rsvc),('lsvc',lsvc)]
clf = VotingClassifier(estimators=estimators, voting="soft")
# self.models = [RandomForestClassifier(n_estimators=100), LDA(), DecisionTreeClassifier()]
models = [rdf, log, lda, knn, lsvc, rsvc, clf]
#models = [rdf]


#################
#### Methods ####
#################

class Params(Resource):
    def get(self, model_id):
      return models[model_id].get_params()

class Evaluation(Resource):
    def get(self):
        return {'hello':'world'}, 201

    def post(self):
        data = request.data
        params = request.args
        classId = params["class"]
        df = pd.read_json(data)
        auto = Automate(models) # Generate a Class
        auto.split_dataframe(df)
        # auto.gridsearchSVM()
        json = {}
        json["crossvalidate"] = auto.get_cross_validation(20) # 20 fold
        #auto.binarize()
        #auto.split_test_train()
        #json["runtime"] = auto.train_models()
        #json["accuracy"] = auto.get_accuracy()
        #json["precision-recall"] = auto.get_precision_recall()
        #json["roc"] = auto.get_roc()
        return json, 201


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response



api.add_resource(Evaluation, '/run')
api.add_resource(Params, '/params/<int:model_id>')

if __name__ == '__main__':
	app.run(debug=True, port=8080)




# tasks = [
#   {
#         'id': 1,
#         'title': u'Buy groceries',
#         'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
#         'done': False
#     },
#     {
#         'id': 2,
#         'title': u'Learn Python',
#         'description': u'Need to find a good Python tutorial on the web', 
#         'done': False
#     }
# ]


# @app.route('/')
# def index():
#   return "Hello, World!"

# @app.route('/todo/api/v1.0/tasks', methods=['GET'])
# def get_tasks():
#     return jsonify({'tasks': tasks})

# # Get the data
# @app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
# def get_task(task_id):
#     task = [task for task in tasks if task['id'] == task_id]
#     if len(task) == 0:
#         abort(404)
#     return jsonify({'task': task[0]})

# # API friendly response
# @app.errorhandler(404)
# def not_found(error):
#   return make_response(jsonify({'error': 'Not found'}), 404)

# # POST command
# @app.route('/todo/api/v1.0/tasks', methods=['POST'])
# def create_task():
#     if not request.json or not 'title' in request.json:
#         abort(400)
#     task = {
#         'id': tasks[-1]['id'] + 1,
#         'title': request.json['title'],
#         'description': request.json.get('description', ""),
#         'done': False
#     }
#     tasks.append(task)
#     return jsonify({'task': task}), 201