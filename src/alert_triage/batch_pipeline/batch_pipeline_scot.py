'''
    This runs a pipeline in batch on scot alerts and then feeds the results
    (i.e. alerts to be examined more closely) back to scot.

'''

import ConfigParser
import time
import logging
import pickle
import json
from stomp import *

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

from alert_triage.feature_extraction import scot_extractor
from alert_triage.feature_extraction.scot_extractor import (SCOTExtractor,
    ALERT_COLLECTION)
from alert_triage.feature_extraction.topic_model_feature_extraction import (
    DatabaseCorpus, TopicModelFeatureExtraction)
from alert_triage.feature_extraction.aggregate_feature_extraction import (
    AggregateFeatureExtraction)
from alert_triage.feature_extraction.raw_feature_extraction import (
    RawFeatureExtraction )
from alert_triage.feature_extraction.cached_alert_correlation import (
    CachedAlertCorrelation)
from alert_triage.feature_extraction.extracted_entities import (
    ExtractedEntities)
from alert_triage.active_learning.qbc import QueryByCommittee

class SCOTBasePipeline(object):
    """ Base class for both training and testing pipelines.
    """

    def __init__(self, configFile, days, limit, pickleFile):
        """ Class constructor.

            Arguments:
            configFile - A .ini file with config parameters.
            days - The number of days over which to grab data.  For the 
                training pipeline, this applies to closed alerts but not
                to promoted alerts.  For the testing pipeline it applies
                to all open alerts.
            limit - Limits the number of total alerts processed by the pipeline.
                A limit of 0 means no limit.
            pickleFile - Where the classifier and other objects should be
                written to (for training pipeline) or read (for testing 
                pipeline).

        """

        # The object that extracts teh features.  It creates a dictionary
        # of alert ids to dictionaries of features.
        self._extractor = None

        # Takes the dictionary of alert ids -> feature dictionaries and 
        # converts that to a vector.
        self._vectorizer = None

        # Takes the vector represenation of the data and trains from the 
        # to create the model (training pipeline) or uses the trainied
        # model to classify instances (testing).
        self._classifier = None

        self._days = days

        self._limit = limit
        if self._limit < 0:
            logging.warn("limit was less than zero, resetting to 0")
            self._limit = 0
        self._pickleFile = pickleFile

        logging.info("limit in base init" + str(self._limit))

        # Read in parameters from config file
        self._config = ConfigParser.RawConfigParser()
        self._config.read(configFile)

	   self._active = self._config.get('ActiveSection', 'feedback')
	   self._budget = self._config.get('ActiveSection', 'budget')

      
    def _extract_all_features(self, featureExtractors, alerts):
        """ Extracts all the features for each alert.

            Arguments:
            featureExtractors - An instance of an AggregateFeatureExtractor
                that has all the desired extractors added to it.
            alerts - A list of all the alerts to process.

            Return:
            Returns a dictionary where the key is the alert id and the value
                is a dictionary of features.
        """

        features = {}
        for alert in alerts:
            alert_features = {}
            alert_features.update(featureExtractors.extract_features(alert))
            if not alert_features:
                logging.debug("No features for alert " + str(alert["id"]))
            features[alert["id"]] = alert_features

        featureExtractors.log_times()

        return features

    def get_classifier(self):
        """ Returns model trained by running the pipeline.
        """
        return self._classifier

    def get_extractor(self):
        """ Returns the extractor
        """
        return self._extractor

    def get_vectorizer(self):
        """ Returns the vectorizer
        """
        return self._vectorizer


 
class SCOTTrainPipeline(SCOTBasePipeline):
    
    def __init__(self, 
                 configFile = "config.ini",
                 days=90,
                 limit=0,
                 pickleFile="classifier.pkl",
                 includeCachedCorrelation=True,
                 includeExtractedEntities=True,
                 includeLDA=True,
                 includeRaw=True):
        """

            Arguments:
            configFile - location of file with configuration parameters
            days - The number of days from today to grab closed alerts.
            limit - Limits the number of alerts.  Tries to balance between
                closed and promoted alerts.
            pickleFile - Where to write the learned classifier.
        """
        super(SCOTTrainPipeline, self).__init__(configFile=configFile,
                                                days=days,
                                                limit=limit,
                                                pickleFile=pickleFile)

        self._includeCachedCorrelation = includeCachedCorrelation
        self._includeExtractedEntities = includeExtractedEntities
        self._includeLDA = includeLDA
        self._includeRaw = includeRaw

            
    def run(self):
        """ 
            Runs the training pipeline.
        """
        # Get the relevant alert ids
        alert_ids, labels = self._get_alert_ids()
        logging.info("Number of alert_ids " + str(len(alert_ids)))

        # Get all the alerts
        alerts = self._get_all_alerts(alert_ids)
        logging.info("Number of alerts " + str(len(alerts)))

        ##### Extract the features #######
        # Create an object that aggregates multiple extractors
        featureExtractors = AggregateFeatureExtraction()
        
        print "self._includeRaw", self._includeRaw
        
        # Add raw feature extractor to the aggregate feature extractor
        if self._includeRaw:
            rawFeatureExtractor = RawFeatureExtraction()
            featureExtractors.add_extractor(rawFeatureExtractor)

        # Add the entity extractor to the aggregate feature extractor
        if self._includeExtractedEntities:
            entityExtractor = ExtractedEntities(self._scot, 
                                                self._config) 
            featureExtractors.add_extractor(entityExtractor)

        # Add cached alert correlation
        # cached alert correlation requires a dictionary mapping alert ids to
        # labels.  We create that in the next line.
        if self._includeCachedCorrelation:
            alert_label_dictionary = self._create_alert_label_dictionary(
                                        alert_ids,labels)
                
            cachedAlertCorrelationExtractor = CachedAlertCorrelation(self._scot,
                                                         alert_label_dictionary) 
            featureExtractors.add_extractor(cachedAlertCorrelationExtractor)
     
        # Add topic model feature extractor
        if self._includeLDA:
            # Needs a corpus object
            corpus = DatabaseCorpus(self._config,
                                    fields=['subject'],
                                    params={"id":alert_ids},
                                    collection=ALERT_COLLECTION,
                                    scot=self._scot)
            lda = TopicModelFeatureExtraction(corpus = corpus,
                                              config = self._config)
            featureExtractors.add_extractor(lda)

     
        # Builds the models for all feature extractors. 
        t1 = time.time()
        featureExtractors.build_model()
        logging.info("METRICS: Time to build models: " + str(time.time() - t1))
        
        # Apply feature extractors to all of the alerts.  Returns a dictionary
        # mapping alert id to a dictionary of features.
        t1 = time.time()
        features = self._extract_all_features(featureExtractors, alerts)
        logging.info("METRICS: Time to extract features: " + 
                     str(time.time() - t1))
        self._extractor = featureExtractors;

        #From the feature dictionary create a matrix
        t1 = time.time()
        X = self._create_feature_matrix(alert_ids, features)
        logging.info("METRICS: Time to create feature matrix: " + 
                     str(time.time() - t1))
        if len(X) >= 10:
            logging.info("First entries of X " + str(X[0:10]))
            logging.info("First entries of labels " + str(labels[0:10]))

        t1 = time.time()
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=1)
        rfc = rfc.fit(X, labels)
        logging.info("METRICS: Time to train classifier: " + 
                     str(time.time() - t1))
        self._classifier = rfc

        logging.info("Score of model " + str(rfc.score(X, labels)))

        # Write the classifier (and the other necessary objects) to disk
        with open(self._pickleFile, 'wb') as outfile:
            self._extractor.pickle(outfile)
            pickle.dump(self._vectorizer, outfile)
            pickle.dump(self._classifier, outfile)


    def _create_alert_label_dictionary(self, alert_ids, labels):
        """ Creates a mapping from alert id to the corresponding label.

            This is required for cached alert correlation.
        """
        mydictionary = {}
        for alert_id, label in zip(alert_ids, labels):
            mydictionary[alert_id] = label
        return mydictionary

    def _create_feature_matrix(self, alert_ids, features):
        """ This creates a matrix that has all the features in it
        """
        features_list = []
        for alert_id in alert_ids:
            features_list.append(features[alert_id])

        self._vectorizer = DictVectorizer(sparse=False)        
        X = self._vectorizer.fit_transform(features_list)
        logging.info("Feature names: " 
                       + str(self._vectorizer.get_feature_names()))

        return X

    def _get_alert_ids(self):
        """ Gets all the alert_ids that we will learn from.

            If limit is set to something greater than 0, we grab some
            of the promoted alerts and closed alerts until the limit is full
            and return that instead of the entire data set.  

            Return:
            Returns the alert ids and labels, both as lists.
        """
        
        labels = []
        promoted_alert_ids = self._get_all_promoted_alert_ids()
        promoted_labels = [1] * len(promoted_alert_ids)
        closed_alert_ids = self._get_closed_alert_ids()
        closed_labels = [0] * len(closed_alert_ids)
        alert_ids = promoted_alert_ids + closed_alert_ids
        labels = promoted_labels + closed_labels 
        
        # if limit is not zero,  
        if self._limit != 0 and len(alert_ids) > self._limit:
            shortened_alert_ids = []
            shortened_labels = []
            for i in range(self._limit):
                # we alternate from either end.  This should give us both
                # closed and promoted events in the filtered set
                if i % 2 == 0:
                    shortened_alert_ids.append(alert_ids[i/2])
                    shortened_labels.append(labels[i/2])
                else:
                    shortened_alert_ids.append(alert_ids[-i/2])
                    shortened_labels.append(labels[-i/2])
            alert_ids = shortened_alert_ids
            labels = shortened_labels
       
        logging.info("Number of alert ids" + str(len(alert_ids)))
        yield alert_ids
        yield labels
        #yield [40686086]
        #yield [1]


    def _get_all_promoted_alert_ids(self):
        """ This gets all promoted alerts regardless of when they occurred
        
            This returns a list of all alert ids that have been promoted,
            i.e. status=promoted.

            Returns:
                The list of alert ids that have been promoted.
        
        """

        params = {"status":"promoted"}

        # This is only one component of the alerts we get, but the total limit
        # is enforced later.  Adding the limit here is to reduce the amount
        # of time it takes to complete the query.
        params['limit'] = self._limit

        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             params)
        alert_ids = []
        for alert in alerts:
            alert_ids.append(alert["id"])

        logging.info("Number of promoted alerts " + str(len(alert_ids)))
        return alert_ids

    def _get_closed_alert_ids(self):
        """ This gets alerts that have been marked closed. 
    
            Returns:
                Returns a list of alert ids in the specified time range
                that are closed.
        """
        logging.info("Entering _get_closed_alert_ids")
        end = int(time.time())
        begin = int(end - self._days * 24*60*60)

        logging.info("begin " + str(begin))
        logging.info("end " + str(end))
        
        params = {}
        params["status"] = "closed"
        params["created"] = [begin, end]
       
        # This is only one component of the alerts we get, but the total limit
        # is enforced later.  Adding the limit here is to reduce the amount
        # of time it takes to complete the query.
        if self._limit != 0:
            params['limit'] = self._limit

        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             params)
        
        alert_ids = []
        if alerts is not None:
            for alert in alerts:
                alert_ids.append(alert["id"])
        
        logging.info("Number of closed alerts: " + str(len(alert_ids))) 
        return alert_ids 
            

    def _get_all_alerts(self, alert_ids):
        """ This gets all the alerts from the specified alert_ids
        """

        params = {}
        params["id"] = alert_ids

        logging.info("Entering _get_all_alerts")
        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             params)

        return alerts



class SCOTTestPipeline(SCOTBasePipeline):
    def __init__(self, 
                 configFile = "config.ini",
                 days=7,
                 limit=0,
                 pickleFile="classifier.pkl"):
        """

            Arguments:
            configFile - location of file with configuration parameters
            days - The number of days from today to grab closed alerts.
            limit - Limits the number of closed alerts.
            pickleFile - Where to read the learned classifier.
        """
        super(SCOTTestPipeline, self).__init__(configFile=configFile,
                                               days=days,
                                               limit=limit,
                                               pickleFile=pickleFile)


    def run(self):
        """ 
            Runs the testing pipeline.
        """
        logging.info("Running the testing pipeline")
        # Read the pickled objects
        with open(self._pickleFile, 'rb') as infile:
            self._extractor = AggregateFeatureExtraction()
            self._extractor.unpickle(infile)
            self._vectorizer = pickle.load(infile)
            self._classifier = pickle.load(infile)

        t1 = time.time()
        open_alerts = self._get_open_alerts()
        logging.info("METRICS: Time to get open alerts: " + 
                     str(time.time() - t1))

        t1 = time.time()
        features = self._extract_all_features(self._extractor, open_alerts)
        logging.info("METRICS: Time to extract features: " + 
                     str(time.time() - t1))

        t1 = time.time()
        X = self._create_feature_matrix(features)
        logging.info("METRICS: Time to create feature matrix: " + 
                     str(time.time() - t1))

        t1 = time.time()
        y = self._classifier.predict_proba(X)
        logging.info("METRICS: Time to predict probabilities: " + 
                     str(time.time() - t1))

        t1 = time.time()
        i = 0
	if self._active != 'False':
	    logging.debug('Running query by committee')
	    budget = int(float(self._budget) * float(len(X)))
	    qbc = QueryByCommittee(budget = budget, model=self._classifier)
	    feedbacked = range(0, len(X))
	    fb = qbc.query(X, y, feedbacked)
	    feedback = dict((i, 0) for i in fb.tolist())
        for i, alert_id in enumerate(features.keys()):
	    logging.debug('Ranking' + " " + str(alert_id) + " " + str(y[i][1]))
	    logging.debug('Probs' + " " + str(alert_id) + " " + str(y[i]))
	    alert_object = {}
	    alert_object['alert_object'] = {}
	    alert_object['alert_object']['triage_probs'] = y[i].tolist()
	    alert_object['alert_object']['triage_ranking'] = y[i][1]
	    if i in feedback and self._active != 'False':
		alert_object['alert_object']['triage_feedback'] = 1
	    else:
	    	alert_object['alert_object']['triage_feedback'] = 0
            try:
                self._scot.modify_alert_field(alert_id, 'alert_object', json.dumps(alert_object['alert_object']))
            except Exception as e:
                logging.exception(str(e))   
            i += 1
        logging.info("METRICS: Time to update triage score: " + 
                     str(time.time() - t1))


    def _get_open_alerts(self):
        params = {}
        end = int(time.time())
        begin = int(end - self._days * 24 * 60 * 60)
	params["created"] = [begin, end]
        params["status"] = "open"
        params["limit"] = self._limit
        alerts = self._scot.query_collection(
                    scot_extractor.ALERT_COLLECTION, params)
        return alerts

    def _create_feature_matrix(self, features):
        """ This creates a matrix that has all the features in it
        """
        features_list = []
        for alert_id in features.keys():
            features_list.append(features[alert_id])

        X = self._vectorizer.transform(features_list)
        logging.info("Feature names: " 
                       + str(self._vectorizer.get_feature_names()))

        return X


class SCOTListenPipeline(SCOTBasePipeline):
    def __init__(self,
                 configFile = "config.ini",
                 pickleFile="classifier.pkl"):
        """
            Arguments:
            configFile - location of file with configuration parameters
            days - The number of days from today to grab closed alerts.
            limit - Limits the number of closed alerts.
            pickleFile - Where to read the learned classifier.
        """
        super(SCOTListenPipeline, self).__init__(configFile=configFile,
					       days=1,
					       limit=0,
                                               pickleFile=pickleFile)

    def run(self):
        """
            Runs the listening pipeline.
        """
        logging.info("Running the listening pipeline")
        # Read the pickled objects
        with open(self._pickleFile, 'rb') as infile:
            self._extractor = AggregateFeatureExtraction()
            self._extractor.unpickle(infile)
            self._vectorizer = pickle.load(infile)
            self._classifier = pickle.load(infile)

        t1 = time.time()

	   self._scotConnection      = self._config.get('ListenerSection', 'connection')
	   self._scotPort		= self._config.get('ListenerSection', 'port')
	   self._scotSubscribe	= self._config.get('ListenerSection', 'subscribe')
	   self._scotAck		= self._config.get('ListenerSection', 'ack')
	   self._listenLength	= self._config.get('ListenerSection', 'hours')

	   self._listen_for_alerts()
	
    def _listen_for_alerts(self):
	class Listener(object):
	    msg = ""
	    def __init__(self):
		self.msg = ""
	    def on_message(self, headers, message):
		self.msg = message

	c = Connection([(self._scotConnection, self._scotPort)])
	listener = Listener()
	c.set_listener('', listener)
	c.start()
	c.connect(wait=True)
	c.subscribe(destination=self._scotSubscribe, id=1, ack=self._scotAck)
	t1 = time.time()
	end = int(t1 + (int(self._listenLength) * 60 * 60))
	msg = ""
	while t1 < end:
	    t1 = time.time()
	    if msg != listener.msg:
	        msg = listener.msg
		alerts = self._message_parser(msg)
		if alerts:
		    for alert in alerts:
		        features = self._extract_all_features(self._extractor, alert)
			X = self._create_feature_matrix(features)
			y = self._classifier.predict_proba(X)
			for i, alert_id in enumerate(features.keys()):
            		    alert_object = {}
            		    alert_object['alert_object'] = {}
            		    alert_object['alert_object']['triage_probs'] = y[i].tolist()
            		    alert_object['alert_object']['triage_ranking'] = y[i][1]
                	    alert_object['alert_object']['triage_feedback'] = 0
            		    try:
                		self._scot.modify_alert_field(alert_id, 'alert_object', json.dumps(alert_object['alert_object']))
            		    except Exception as e:
                		logging.exception(str(e))
	c.disconnect()
	

    def _message_parser(self, msg):
	   """ This takes the messages from ActiveMQ and extracts  the
	   individual alerts.
	   """
	   alerts = None
	   msg = json.loads(msg)
	   if "action" not in msg:
	       return alerts
	   if msg["action"] != "created" and msg["action"] != "updated":
	       return alerts
	   msg_type = msg["data"]["type"] 
       msg_id = msg["data"]["id"]
	   params = {}
	   params["id"] = msg_id
	   if msg_type == "alert":
	       extract = scot_extractor.ALERT_COLLECTION
	   elif msg_type == "entity":
	       extract = scot_extractor.ENTITY_COLLECTION
	   elif msg_type == "event":
	       extract = scot_extractor.EVENT_COLLECTION
	   elif msg_type == "alertgroup":
	       extract = scot_extractor.ALERT_GROUP_COLLECTION
	   elif msg_type == "incident":
	       extract = scot_extractor.INCIDENT_COLLECTION
	   elif msg_type == "entry":
	       extract = scot_extractor.ENTRY_COLLECTION
	   else:
	       return alerts
        alerts = self._scot.query_collection(extract, params)
	   return alerts

    def _create_feature_matrix(self, features):
        """ This creates a matrix that has all the features in it
        """
        features_list = []
        for alert_id in features.keys():
            features_list.append(features[alert_id])

        X = self._vectorizer.transform(features_list)
        logging.info("Feature names: "
                       + str(self._vectorizer.get_feature_names()))

        return X
