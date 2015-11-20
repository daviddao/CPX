#!/usr/bin/env python
#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""WebSocket Tornado Server handeling requests from CellProfiler Luminosity
"""

import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import tornado.httpserver
import os.path
import uuid

from tornado.options import define, options

import pandas as pd
import time

### MODELS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn.svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
###

### VALIDATION
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
###


define("port", default=8002, help="run on the given port", type=int)

def crazy():
    import pandas as pd

    import time

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

    # Normalize and scale the dataset


    df = pd.read_csv('./hela_norm.csv')
    y = df.pop("Class")

    y = y.values
    X = df.values

    folds = 20
    n = len(y)
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
    # rbf_clf = grid_search.GridSearchCV(SVC(probability=True),param_grid={'gamma': [2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**0, 2**1], 'C': [2**-1, 2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]},cv=grid_cv,n_jobs=-1)

    # models = [linear_clf, rbf_clf]
    ## TEST END

    # Get accuracy
    def scorer(model, X, y):
        return model.score(X, y)

    def name(model):
        return model.__class__.__name__

    for model in models:

        model_name = name(model)
        print "Now getting CV scores for", model_name
        start = time.time()

        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)

        runtime = time.time() - start
        print('cv-accuracy for {}: {} +- {} and {} seconds'.format(model_name, scores.mean(), scores.std(), runtime))
        d = {}
        d["name"] = name(model)
        d["time"] = "{0:.5f}".format(runtime)
        d["score"] = "{0:.3f}".format(scores.mean())
        d["std"] = "+-{0:.5f}".format(scores.std())
        json_cv_time.append(d)

    print json_cv_time

# The model for the Tornado Web Application
class Model():


    # Dictionary with supported sklearn models (and more in the future)

    ##############
    ### Models ###
    ##############

    rdf = RandomForestClassifier(n_estimators=100)
    ada = OneVsRestClassifier(AdaBoostClassifier())
    lsvc = LinearSVC()
    rsvc = SVC(probability=True, C=32, gamma=0.125) # tuned HP for HELA
    log = LogisticRegression()
    lda = LDA()
    knn = KNeighborsClassifier()
    grad = GradientBoostingClassifier()

    models = {
        "RandomForestClassifier" : rdf,
        "AdaBoostClassifier" : ada,
        "LinearSVC" : lsvc,
        "SVC" : rsvc,
        "LogisticRegression" : log,
        "LinearDiscriminantAnalysis" : lda,
        "KNeighborsClassifier" : knn,
        "GradientBoostingClassifier" : grad
    }

    trained_models = [] # List of models we trained

    trainingSet = None # List of TrainingSet

    @classmethod
    def updateTrainingSet(cls, df):
        # Some preprocessing
        y = df.pop("Class")
        y = y.values
        X = df.values

        cls.trainingSet = {}
        cls.trainingSet["X"] = X
        cls.trainingSet["y"] = y

    @classmethod
    def getTrainingSet(cls):
        return cls.trainingSet

    @classmethod
    def getParams(cls, classifierName):
        return cls.models[classifierName].get_params()

    @classmethod
    def setParams(cls, classifierName, params):
        cls.models[classifierName].set_params(**params)

    @classmethod
    def getCV(cls, splits):
        
        n = len(cls.trainingSet["y"])
        cv = KFold(n, n_folds=splits, shuffle=True)

        return cv

        




class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/classifiersocket", ClassifierSocketHandler),
            (r"/csv", CSVHandler),
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,
        )
        tornado.web.Application.__init__(self, handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", messages=ClassifierSocketHandler.cache)

from tornado_cors import CorsMixin
class CSVHandler(CorsMixin, tornado.web.RequestHandler):

    # Value for the Access-Control-Allow-Origin header.
    # Default: None (no header).
    CORS_ORIGIN = '*'

    # Value for the Access-Control-Allow-Headers header.
    # Default: None (no header).
    CORS_HEADERS = 'Content-Type'

    # Value for the Access-Control-Allow-Methods header.
    # Default: Methods defined in handler class.
    # None means no header.
    CORS_METHODS = 'POST'

    # Value for the Access-Control-Allow-Credentials header.
    # Default: None (no header).
    # None means no header.
    CORS_CREDENTIALS = True

    # Value for the Access-Control-Max-Age header.
    # Default: 86400.
    # None means no header.
    CORS_MAX_AGE = 21600

    # Value for the Access-Control-Expose-Headers header.
    # Default: None
    CORS_EXPOSE_HEADERS = 'Location, X-WP-TotalPages'

    def post(self):
        data_json = self.request.body
        df = pd.read_json(data_json) # Pandas parses the json data into a dataframe
        Model.updateTrainingSet(df)

    def get(self):
        trainingSet = Model.getTrainingSet()
        # logging.info(trainingSet)
    
# Controller ? 
class ClassifierSocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()
    cache = []
    cache_size = 200

    # Allow cross-origin
    def check_origin(self, origin):
        return True

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        ClassifierSocketHandler.waiters.add(self)

    def on_close(self):
        ClassifierSocketHandler.waiters.remove(self)

    @classmethod
    def update_cache(cls, chat):
        cls.cache.append(chat)
        if len(cls.cache) > cls.cache_size:
            cls.cache = cls.cache[-cls.cache_size:]

    @classmethod
    def send_updates(cls, chat):
        logging.info("sending message to %d waiters", len(cls.waiters))
        for waiter in cls.waiters:
            try:
                waiter.write_message(chat)
            except:
                logging.error("Error sending message", exc_info=True)

    def on_message(self, message):
        logging.info("got message %r", message)
        parsed = tornado.escape.json_decode(message)

        # Check for which request was made , bad code - maybe replace with switch-case
        get_params = parsed[0]["get_params"]
        params = None
        if get_params:
            params = Model.getParams(parsed[0]["classifier"])
            logging.info(params)
            ClassifierSocketHandler.send_updates(params)

        set_params = parsed[0]["set_params"]
        if set_params:
            params = parsed[0]["params"]
            Model.setParams(parsed[0]["classifier"],params)

        train = parsed[0]["train"]
        if train:
            classifierName = parsed[0]["classifier"]
            splits = parsed[0]["splits"]
            classifier = Model.models[classifierName]
            X = Model.trainingSet["X"]
            y = Model.trainingSet["y"]
            cv = Model.getCV(splits) # Get splits

            logging.info("Starting Crossvalidation with %d splits", splits)

            start = time.time()
            # Just to know what we are dealing with here
            def scorer(classifier, X, y):
                return classifier.score(X, y)

            scores = cross_val_score(classifier, X, y, cv=cv, scoring=scorer)
            runtime = time.time() - start

            logging.info('cv-accuracy for {}: {} +- {} and {} seconds'.format(classifierName, scores.mean(), scores.std(), runtime))
            d = {}
            d["name"] = classifierName
            d["time"] = "{0:.5f}".format(runtime)
            d["score"] = "{0:.3f}".format(scores.mean())
            d["std"] = "+-{0:.5f}".format(scores.std())
            d["params"] = Model.getParams(classifierName)
            d["splits"] = splits
            json_cv_time = d
            logging.info(d)
            ClassifierSocketHandler.send_updates(json_cv_time)


        # chat = {
        #    "id": str(uuid.uuid4()),
        #     "body": parsed["body"],
        #     }
        # chat["html"] = tornado.escape.to_basestring(
        #     self.render_string("message.html", message=chat))

        # # logging.info("Yeah lets get started!")
        # # crazy()

        # ClassifierSocketHandler.update_cache(chat)
        # ClassifierSocketHandler.send_updates(chat)


def main():
    tornado.options.parse_command_line()
    app = Application()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()