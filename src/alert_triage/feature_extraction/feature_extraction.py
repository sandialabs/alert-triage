"""Extract Features.

This class creates instances for all the classes that extract features
and then utilizes these instances to extract features.  After running,
the features instance variable of the FeatureExtraction class is a
dictionary containing all the features that were extracted.

FeatureExtractionException: this is the exception type that should be
raised when exceptions or errors occur.

FeatureExtraction: this class extracts features from a single alert
using raw features and the extracted entities collection.

"""

import time
import logging

from alert_triage.feature_extraction import (cached_alert_correlation
    extracted_entities, label_extraction, raw_feature_extraction
    topic_model_feature_extraction, scot_extractor)


class FeatureExtractionException(Exception):

    """Exception type for the FeatureExtraction Class"""

    pass




class FeatureExtraction(object):

    """Extract Features.

    This class wraps the other classes that extract features from
    various sources.

    extract_features():  this function extracts features.

    features: this instance variable contains the dictionary of
    features after extract_features has been run.  The key for the
    dictionary is the alert_id.  The dictionary is similar to a sparse
    matrix.

    """

    def __init__(self, scot, labels):
        """Create an instance of the FeatureExtraction class.

        Arguments:
            scot - an interface to the scot rest api
            labels - This is a dictionary where the key are alert ids and the
                values are the labels of the alerts.  This comes from
                LabelExtraction.extract_labels.

        """
        self._scot = scot
        self._labels = labels
        self.features = {}
        self._extractors = []
        #self._set_up_objects()
        # TODO: This doesn't make sense: first it populates the unique values
        # and then they are wiped out the next line
        #self._populate_unique_values()
        self._unique_values = {}
    
    def _populate_unique_values(self):
        """Populate a dictionary of all unique values"""
        logging.info( "Populating unique values" )
        features_counts = {}
        unique_values = {}
        alert_ids = [key for key, val in self._labels.iteritems()]
        logging.info("alert_ids length " + str(len(alert_ids)))
        params = {"id":alert_ids}
        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION, 
                                             params)

        for alert in alerts:
            for extractor in self._extractors:
                features = extractor.extract_features(alert)
                if features:
                    feature_dict = features
                    for key, value in feature_dict.iteritems():
                        feature_string = '_'.join([unicode(i) for i in 
                                                  (key, value)])
                        if (feature_string in features_counts and 
                            isinstance(value, basestring)):
                            features_counts[feature_string] += 1
                        elif isinstance(value, basestring):
                            features_counts[feature_string] = 1
        for feature in features_counts:
            if features_counts[feature] == 1:
                unique_values[feature] = features_counts[feature]
	    logging.info("Finished populating unique values")
        self._unique_values = unique_values

    def add_default_extractors(self, scot, labels):
        """ This adds default extractors to the FeatureExtraction object

            This adds a set of extractors to the FeatureExtraction object.
            The default extractors add are:

            1) RawFeatureExtractor
            2) ExtractedEntities
            3) TopicModelFeatureExtractor
            4) CachedAlertCorrelation

            Arguments:
                extractors - A list of extractors
                scot - An interface to the scot REST api
                labels - Alerts with labels
        """
        #### RawFeatureExtractor ######
        self.add_extractor(raw_feature_extraction.RawFeatureExtraction())
        #self._populate_unique_values()
        
        #### ExtractedEntities ########
        self.add_extractor(extracted_entities.ExtractedEntities(
                                    self.scot))

        ##### TopicModelFeatureExtractor(LDA) #####
        labeled_alert_ids = label_extraction.labeled_ids(self._labels)
        labeled_alert_query = {"alert_id": labeled_alert_ids}

        # Corpus object needed by TopicModelFeatureExtractions
        corpus = topic_model_feature_extraction.DatabaseCorpus(
                    scot=self._scot,
                    fields=["subject"],
                    params=labeled_alert_query)

        self.add_extractor(
            topic_model_feature_extraction.TopicModelFeatureExtraction(
                corpus=corpus, iterations=100))

        # TODO: Not sure we are going to use this
        #self._extractors.append(raw_feature_extraction.
        # SplunkSearchFeatureExtraction(
        #    database_name=self._alert_database.database))

        ##### CachedAlertCorrelation #####
        self.add_extractor(cached_alert_correlation.CachedAlertCorrelation(
            self._scot, self._labels))
       
        # If an extractor needs a model to be built before features are
        # extracted, this takes care of it. 
        for extractor in self._extractors:
            logging.info(
                "Building model for {0} . . .".format(str(extractor)))
            extractor.build_model()




    def add_extractor(self, extractor):
        """ Adds the specified extractor to the list of extractors.

            Before extract_features is called, this should be called
            for each extractor that is to be included in the feature
            extraction process.

            Arguments:
                extractor The extractor object to add.

            Returns:
                Doesn't return anything.
        """
        self._extractors.append(extractor)

    """
        Initialize the feature extractors.  The feature extractors that
        are created are
        1) RawFeatureExtractor
        2) ExtractedEntities
        3) TopicModelFeatureExtractor
        4) CachedAlertCorrelation
    """


    def extract_features(self):
        """Extract features."""
        unlabeled_alert_ids = label_extraction.unlabeled_ids(self._labels)
        logging.info("Projecting alerts through extractors . . .")
        timings = {}
        for extractor in self._extractors:
            timings[str(extractor)] = 0.0

        params = {"id": unlabeled_alert_ids}


        # This could be done better with batches
        for alert_id in unlabeled_alert_ids:
            params = {"id": alert_id}
            alert = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                                params)

            alert_id = alert["id"]
            feature_dict = {}
            update_dict = {}
            for extractor in self._extractors:
                start = time.time()
                features = extractor.extract_features(alert)
                if features:
                    feature_dict.update(features)
                    for feature in feature_dict:
                        if not isinstance(feature_dict[feature], basestring):
                            update_dict[feature] = feature_dict[feature]
                        elif (str(feature)+"_"+str(feature_dict[feature]) 
                              not in self._unique_values):
                            update_dict[str(feature)+"_"+str(
                                feature_dict[feature])] = 1
                timings[str(extractor)] += time.time() - start
            if len(update_dict) > 0:
                self.features[alert_id] = update_dict
            if index % 10000 == 0 and index != 0:
                logging.info("Alerts processed: {0}".format(index + 1))
            for extractor, total_time in timings.iteritems():
                logging.info( "Extractor: {0}, Total time: {1}, Alerts per" +
                              " second: {2}".format(
                    extractor, total_time, len(self.features)/total_time))
        return self.features
