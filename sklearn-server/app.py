#!flask/bin/python

from automate import Automate

from flask import Flask, jsonify, make_response, request
from flask_restful import Resource, Api, reqparse # Flask API
import pandas as pd 


app = Flask(__name__)
api = Api(app) # Generate an API

# models



class Evaluation(Resource):
    def get(self):
        return {'hello':'world'}, 201

    def post(self):
        data = request.data
        df = pd.read_json(data)
        #print df.head()
        auto = Automate() # Generate a Class
        auto.split_dataframe(df)
        auto.split_train_test()
        auto.init_models()
        json = {}
        json["accuracy"] = auto.get_accuracy()
        return json, 201


@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response



api.add_resource(Evaluation, '/run')

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