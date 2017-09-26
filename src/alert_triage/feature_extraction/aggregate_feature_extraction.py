"""Extract Features.

This class aggregates many different extractors together and extracts features
using all of the extractors.

FeatureExtractionException: this is the exception type that should be
raised when exceptions or errors occur.


"""

import time
import logging
import pickle

from alert_triage.feature_extraction import (cached_alert_correlation,
    extracted_entities, label_extraction, raw_feature_extraction,
    topic_model_feature_extraction, scot_extractor)
from alert_triage.feature_extraction.abstract_feature_extraction import (
    AbstractFeatureExtraction)
from alert_triage.feature_extraction.topic_model_feature_extraction import (
    TopicModelFeatureExtraction)
from alert_triage.feature_extraction.raw_feature_extraction import (
    RawFeatureExtraction)
from alert_triage.feature_extraction.cached_alert_correlation import (
    CachedAlertCorrelation)
from alert_triage.feature_extraction.extracted_entities import (
    ExtractedEntities)

class AggregateFeatureExtractionException(Exception):

    """Exception type for the AggregateFeatureExtraction Class"""

    pass




class AggregateFeatureExtraction(AbstractFeatureExtraction):

    """Extract Features.

    This class aggregates the other classes that extract features from
    various sources.

    extract_features():  this function extracts features.

    build_model(): this function calls build_model on all registered extractors.

    add_extractor(): this adds an extractor to the list of extractors.

    """

    def __init__(self):
        """Create an instance of the AggregateFeatureExtraction class.

        Arguments:

        """
        self._extractors = []

        # Keeps track of how much time each extractor takes (for metric 
        # purposes).
        self._timePerExtractor = {}

    def pickle(self, handle):
        pickle.dump(len(self._extractors), handle)
        for extractor in self._extractors:
            pickle.dump(type(extractor), handle)
            extractor.pickle(handle)
        #pickle.dump(self._timePerExtractor, handle)
        
    def unpickle(self, handle):
        numExtractors = pickle.load(handle)
        for i in range(numExtractors):
            # Don't really like this.  Aggregate shouldn't need to know 
            # about all the extractor classes.  However, because of the pickle
            # problem with ExtractedEntities, this is a work around.  Mabye
            # there is a nother work around that avoids this cludge.
            typeinfo = pickle.load(handle)
            if typeinfo is CachedAlertCorrelation:
                extractor = CachedAlertCorrelation()
            elif typeinfo is ExtractedEntities:
                extractor = ExtractedEntities()
            elif typeinfo is RawFeatureExtraction:
                extractor = RawFeatureExtraction()
            elif typeinfo is TopicModelFeatureExtraction:
                extractor = TopicModelFeatureExtraction()
            extractor.unpickle(handle)
            self.add_extractor(extractor)
        #self._timePerExtractor = pickle.load(handle) 

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
        self._timePerExtractor[extractor] = 0


    def build_model(self):
        """ Builds the models for each registered extractor
        """
        for extractor in self._extractors:
            beg = time.time()
            extractor.build_model()
            logging.info("Time to build model for extractor " + str(extractor) +
                         ": " + str(time.time() - beg))

    def extract_features(self, alert):
        """ Extract features.
        
            Iterate over all the extractors that have been added to the list
            and applies them to the specified alert.

            Arguments:
                alert - A dictionary representing an alert.

            Return:
                Returns a dictionary where the key is the feature name and the
                value is the feature value.
        """
        print "extractors length:", len(self._extractors)
        features = {}
        for extractor in self._extractors:
            logging.debug("Applying extractor " + str(extractor) + " to "+
                          "alert " + str(alert.get("id")))
            beg = time.time()
            featureUpdate = None
            try:
                featureUpdate = extractor.extract_features(alert)
            except Exception as e:
                logging.exception(str(e))
            if featureUpdate is not None:
                features.update(featureUpdate)
            self._timePerExtractor[extractor] += time.time() - beg

        return features

    def log_times(self):
        """ Adds to the log the time taken by each extractor.
        """
        for extractor in self._extractors:
            logging.info("METRICS: Time for " + str(extractor) + ": " +
                         str(self._timePerExtractor[extractor]))
