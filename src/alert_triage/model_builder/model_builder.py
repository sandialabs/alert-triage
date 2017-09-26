"""

This class builds the model from labels and features for
Alert Triage LDRD.

"""

import numpy
from sklearn import tree
from sklearn import cross_validation
from sklearn import svm

_DEBUG = True

class ModelBuilder(object):
    def __init__(self, database, collection_name='extractedFeatures', labels={},
                 model='tree', cross_validation=True):
        self._database = database
        self._collection_name = collection_name
        self._labels = labels
        self._model = model
        self._cross_validation = cross_validation
        self.features = []
        self.class_labels = []


    def process_data(self):
        feature_list_cursor = self._database.find({"all_features":{"$exists":True}},
                                                   collection=self._collection_name)
        feature_list = feature_list_cursor[0].get("all_features")

        for alert_id in self._labels:
            self.class_labels.append(self._labels[alert_id])
            temp_list = [0 for _ in feature_list]
            features_cursor = self._database.find({"alert_id":alert_id},
                                                   collection=self._collection_name)
            temp_features = features_cursor[0]
            del temp_features["_id"]
            del temp_features["alert_id"]
            for feature in temp_features:
                temp_list[feature_list.index(feature)] = temp_features[feature]
            self.features.append(temp_list)


    def build_model(self):
        if _DEBUG:
            print "processing data for model"
        self.process_data()

        if self._model == 'tree':
            if _DEBUG:
                print "Fitting Model"
            self.fit_model = tree.DecisionTreeClassifier()
        if self._model == 'svm':
            if _DEBUG:
                print "fitting SVM"
            self.fit_model = svm.SVC()
        if self._cross_validation:
            if _DEBUG:
                print "len(self.features): {0}".format(len(self.features))
                print "len(self.class_labels): {0}".format(len(self.class_labels))
            self.scores = cross_validation.cross_val_score(self.fit_model,
                                                           self.features,
                                                           numpy.array(self.class_labels),
                                                           cv=5)
            if _DEBUG:
                print self.scores
        else:
            self.fit_model = self.fit_model.fit(self.features, numpy.array(self.class_labels))
