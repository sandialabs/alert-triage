"""Run the batch pipeline.

This class executes the batch processing pipeline for label
extraction, feature extraction, model building, and other steps.  It
may update the backend database that stores labels, features, models,
and other artificats.  It may also send the rankings and queries to
SCOT.

BatchPipelineException: this is the exception type that should be
raised when exceptions or errors occur.

BatchPipeline: this is the only class defined in this module.  Its
run_pipeline method does the actual work of running the pipeline.

To Do:

"""

import json
import pickle
import time

import os
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

import alert_triage
from alert_triage.database import database
from alert_triage.feature_extraction import feature_extraction
from alert_triage.feature_extraction import label_extraction
from alert_triage.active_learning import uncertainty
from alert_triage.util import scot_helper

_DEBUG = True

_ALERT_IDS_PICKLE_FILE = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/alert_ids.pkl")
_FEATURE_PICKLE_FILE = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/features.pkl")
_LABEL_PICKLE_FILE = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/labels.pkl")
_MODEL_PICKLE_FILE = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/rfc.pkl")
_LABELED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/")
_PROB_LABELS = ["closed", "event", "incident"]


class BatchPipelineException(Exception):

    """Exception type for the Batch Pipeline Class"""

    pass


class BatchPipeline(object):

    """Run the batch pipeline.

    This class runs the batch pipeline for various configurations.
    Currently, the only two configurations are test and normal.

    run_pipeline(): this method runs the batch pipeline.

    """

    def __init__(self, testing=False):
        """Create an instance of the BatchPipeline class.

        Arguments:
            testing:
                A boolean that if true indicates that we are testing
                the pipeline.  Currently this simply limits the number
                of alerts from which we extract features.

        """
        self._testing = testing
        self._alert_id_map = {}
        self._json_dict = {}
        self._alert_rankings = {}
        self._unlabeled_alert_ids = []
        # Automatically set up class weights.
        weights = numpy.array([value for value in
                               scot_helper.LABEL_WEIGHTS.itervalues()
                               if value is not None])
        self._class_weights = numpy.unique(weights).astype(float)/numpy.max(weights).astype(float)

        if _DEBUG:
            print "Instance variables:"
            for name, value in self.__dict__.items():
                print "    {0}: {1}".format(name, value)

    def run_pipeline(self):
        """Run the batch pipeline."""
        start_time = time.time()
        if _DEBUG:
            print "Setting up the alerts databases . . ."
        alert_database = database.Database(database=scot_helper.SCOT_DATABASE)

        # Limit the number of alerts if this is a test.
        if self._testing:
            limit = scot_helper.NUM_ALERTS_TESTING
        else:
            limit = scot_helper.ALL_ALERTS

        # Wipe out the "triage_feedback" flag of alerts that were requested to
        # be labeled, but never were.
        if not self._testing:
            if _DEBUG:
                print "Resetting alerts to query. . ."
            cursor = alert_database.find({"triage_feedback": 1,
                                          "status": "open"})
            feedback_json = {"triage_feedback": 0}
            for alert in cursor:
                alert_id = alert["alert_id"]
                print alert
                #scot_helper.scot_put(alert_id, feedback_json)

        # Extract the labels for the alerts.
        if _DEBUG:
            print "Extracting labels . . ."
        cursor = alert_database.find().sort("created", -1).limit(1)
        most_recent_entry = 0
        for alert in cursor:
            most_recent_entry = alert['created']
        label_extractor = label_extraction.LabelExtraction(
            database=alert_database,
            date=scot_helper.seconds_epoch(most_recent_entry, 3),
            limit=limit,
            weights=scot_helper.LABEL_WEIGHTS)
        labels = label_extractor.extract_labels()

        if _DEBUG:
            print "Length of label dict: {0}".format(len(labels))

        # Extract features.
        if _DEBUG:
            print "Extracting features . . ."

        # Joe is working on the feature extraction class.  He's adding
        # other feature types to the extract_features method and he'll
        # also save models (e.g., LDA) used in feature
        # extraction.
        feature_extractor = feature_extraction.FeatureExtraction(
            alert_database=alert_database,
            labels=labels,
            )
        feature_extractor.extract_features()
        alert_ids = []
        features = []
        for key, value in feature_extractor.features.iteritems():
            alert_ids.append(key)
            features.append(value)

        if _DEBUG:
            print "Finding indices . . ."
        (unlabeled_indices, labeled_indices, y) = self._find_labeled(labels,
                                                                     alert_ids)

        if _DEBUG:
            numerator = numpy.bincount(y[labeled_indices].astype(int)).astype(float)
            print "Label distribution: {0}".format(numerator/float(len(labeled_indices)))
            print "Unlabeled: {0}, Labeled: {1}".format(len(unlabeled_indices),
                                                        len(labeled_indices))

        # Now that we've extracted the features, vectorize and create
        # your X and y to use to build the models.
        if _DEBUG:
            print "Vectorizing . . ."
        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(features)

        if _DEBUG:
            print "len(feature names): {0}".format(X.shape[1])

        # Select features.  We could do a model-agnostic feature
        # selection and try to select some generally optimal set.  Or,
        # we can select features that will work best for a specific
        # model, e.g., random forest.

        # Here is where we will likely evaluate the models to
        # determine which one to use for the actual ranking.  It is
        # also where we will evaluate the active learning algorithms.
        # Uh, not sure about that.  It seems like we've moved the
        # evaluation elsewhere to not slow down the pipeline.

        # Build and save the model.  Use it to predict the class of
        # all unlabeled alerts.
        if _DEBUG:
            print "Building model . . ."
            print "Training matrix shape: {0}".format(X[labeled_indices, :].shape)
        # With n_jobs=-1, the number of threads will be set to the
        # number of cores.
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rfc = rfc.fit(X[labeled_indices, :], y[labeled_indices])

        if _DEBUG:
            print "Saving data to use in evaluation . . ."
        numpy.save(_LABELED_DATA_PATH + "X.npy", X[labeled_indices, :])
        numpy.save(_LABELED_DATA_PATH + "y.npy", y[labeled_indices])
        outfile = open(_LABELED_DATA_PATH + "feature_names.pkl", "wb")
        pickle.dump(vectorizer.get_feature_names(), outfile)
        outfile.close()

        if not self._testing:
            if _DEBUG:
                print "Saving alert ids. . ."
            pickle_file = open(_ALERT_IDS_PICKLE_FILE, "wb")
            pickle.dump(alert_ids, pickle_file, -1)
            pickle_file.close()

            if _DEBUG:
                print "Saving features. . ."
            pickle_file = open(_FEATURE_PICKLE_FILE, "wb")
            pickle.dump(features, pickle_file, -1)
            pickle_file.close()

            if _DEBUG:
                print "Saving labels. . ."
            pickle_file = open(_LABEL_PICKLE_FILE, "wb")
            pickle.dump(labels, pickle_file, -1)
            pickle_file.close()

            if _DEBUG:
                print "Saving model . . ."
            pickle_file = open(_MODEL_PICKLE_FILE, "wb")
            pickle.dump(rfc, pickle_file, -1)
            pickle_file.close()

        #if _DEBUG:
        #    print "Predicting . . ."
        #predictions = rfc.predict(X[unlabeled_indices, :])

        # Need to get the probabilities also.
        if _DEBUG:
            print "Obtaining probabilities . . ."
        probs = rfc.predict_proba(X[unlabeled_indices, :])

        # Perform active learning.
        if _DEBUG:
            print "Performing active learning . . ."
        active_learner = uncertainty.UncertaintySampling(budget=6, model=rfc)
        query_indices = active_learner.query(X, y, unlabeled_indices)

        if _DEBUG:
            print "query indices: {0}".format(query_indices)

        # Convert query_indices to alert_ids.
        query_alert_ids = self._convert_indices_to_ids(query_indices)
        if _DEBUG:
            print "query alert ids: {0}".format(query_alert_ids)

        # Create a list of tuples of the form (alert_id, alert_json).
        # alert_json contains the ranking, the probabilities, and the
        # query status of alert alert_id.
        if _DEBUG:
            print "Creating the list of alert_ids and json objects . . ."
        self._build_json(probs, query_alert_ids)

        # Determine which rankings to send to SCOT.
        if _DEBUG:
            print "Determining which rankings to send to SCOT . . ."
        rankings_to_send = self._pick_rankings_to_send(query_alert_ids)

        # Use Josh's code to send rankings to SCOT.
        if _DEBUG:
            print "Sending alerts to SCOT if not in testing mode."
        if not self._testing:
            for alert_id, json_data in rankings_to_send:
                scot_helper.scot_put(alert_id, json_data)

        # Teardown.
        alert_database.close()

        print "Total time (in seconds):", time.time() - start_time

    def _pick_rankings_to_send(self, query_alert_ids):
        """Pick which rankings to send to SCOT and save in the db."""

        rankings_db = database.Database(database=scot_helper.SCOT_DATABASE,
                                        collection=_RANKINGS_COLLECTION)

        # Construct a numpy array of the differences.
        ids_and_diffs = {}
        ranking_diffs = []
        new_alerts = []
        for alert_id, new_ranking in self._alert_rankings.iteritems():
            old_ranking = rankings_db.get_last_sent_ranking(alert_id)
            if old_ranking is None:
                new_alerts.append(alert_id)
            else:
                diff = new_ranking - old_ranking
                ids_and_diffs[alert_id] = diff
                ranking_diffs.append(diff)

        ranking_diffs = numpy.array(ranking_diffs)

        # Only send in rankings whose difference is greater than or
        # equal to one standard deviation above the mean.
        if len(ranking_diffs) == 0:
            print "There are no ranking differences."
        else:
            mean = ranking_diffs.mean()
            std_dev = ranking_diffs.std()
            if _DEBUG:
                print "mean of ranking diffs: {0}".format(mean)
                print "std dev of ranking diffs: {0}".format(std_dev)
        filtered_diffs = []
        for alert_id, diff in ids_and_diffs.iteritems():
            if (diff - mean) >= std_dev:
                filtered_diffs.append(alert_id)

        # If the difference of the rankings passed the above filter,
        # it now has to pass another filter.  This time, the actual
        # value of the ranking, not the difference, must be greater
        # than or equal to one standard deviation above the mean.
        rankings = numpy.array(self._alert_rankings.values())
        mean = rankings.mean()
        std_dev = rankings.std()
        if _DEBUG:
            print "mean of rankings: {0}".format(mean)
            print "std dev of rankings: {0}".format(std_dev)
        rankings_to_send = []
        # If an alert is new, meaning we don't have a previous ranking
        # for it, or it is an alert for which we are querying the
        # analysts for a label, send it in no matter what.
        query_new_set = set(new_alerts + query_alert_ids)
        for alert_id in query_new_set:
            rankings_to_send.append((alert_id, self._json_dict[alert_id]))
            if not self._testing:
                ranking = self._alert_rankings[alert_id]
                rankings_db.add_ranking(alert_id, ranking)
        # Also send in rankings that pass the difference and ranking
        # criteria.
        for alert_id in set(filtered_diffs) - set(query_alert_ids):
            ranking = self._alert_rankings[alert_id]
            if (ranking - mean) >= std_dev:
                rankings_to_send.append((alert_id, self._json_dict[alert_id]))
                if not self._testing:
                    rankings_db.add_ranking(alert_id, ranking)

        if _DEBUG:
            print "# of rankings to send: {0}".format(len(rankings_to_send))

        rankings_db.close()

        return rankings_to_send

    def _convert_indices_to_ids(self, indices):
        """Convert indices to alert ids."""
        alert_ids = []
        for index in indices:
            alert_ids.append(self._alert_id_map[index])
        return alert_ids

    def _build_json(self, probabilities, query_alert_ids):
        """Create a dictionary of alert JSON objects to send to SCOT."""
        for alert_id, prob_vector in zip(self._unlabeled_alert_ids, probabilities):
            alert_object = {}
            ranking = numpy.dot(prob_vector, self._class_weights)
            ranking = ranking * _RANKING_MULTIPLIER
            alert_object["triage_ranking"] = ranking
            self._alert_rankings[alert_id] = ranking
            if alert_id in query_alert_ids:
                alert_object["triage_feedback"] = 1
            else:
                alert_object["triage_feedback"] = 0
            probs = {}
            for index, class_prob in enumerate(prob_vector):
                probs[_PROB_LABELS[index]] = class_prob
            alert_object["triage_probs"] = probs
            if _DEBUG and alert_object["triage_feedback"] == 1:
                print "alert_object:\n{0}".format(alert_object)
            json_object = json.dumps(alert_object)
            self._json_dict[alert_id] = json_object

    def _find_labeled(self, labels, alert_ids):
        """Iterate through all the labels and find labeled/unlabeled indices."""
        unlabeled_indices = []
        labeled_indices = []
        y = []
        for index, alert_id in enumerate(alert_ids):
            label = labels.get(alert_id, None)
            if label is None:
                unlabeled_indices.append(index)
                self._unlabeled_alert_ids.append(alert_id)
                y.append(-1)
            else:
                labeled_indices.append(index)
                y.append(label)
            self._alert_id_map[index] = alert_id
        return (numpy.array(unlabeled_indices), numpy.array(labeled_indices),
                numpy.array(y))
