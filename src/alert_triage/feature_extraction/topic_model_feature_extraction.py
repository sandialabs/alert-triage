"""Extract features using topic modeling
(e.g., Latent Dirichlet Allocation (LDA))

The TopicModelFeatureExtraction class operates on a specific field in an
alert (e.g., subject).  It will build the LDA model on all alerts
in the database that contain this field.

This class uses an external Python library (gensim), with an LDA implementation
based on the following paper:
Matthew Hoffman, David M. Blei, Francis Bach. Online Learning
for Latent Dirichlet Allocation. NIPS (2010).

More info on LDA and topic modeling:
http://www.cs.princeton.edu/~blei/topicmodeling.html

TopicModelFeatureExtractionException: exception type that should be
raised when exceptions or errors occur.
TopicModelFeatureExtraction: a model-based feature extractor that uses
latent Dirichlet allocation.  Extract features from an alert using the
extract_features() method.

"""

import pickle
from gensim import corpora
from gensim.models import ldamodel
from os import path
import alert_triage
from alert_triage.database import database
from alert_triage.feature_extraction.abstract_feature_extraction import (
    AbstractFeatureExtraction)

class TopicModelFeatureExtractionException(Exception):

    """Exception type for the LDAFeatureExtraction class."""

    pass


class DatabaseCorpus(object):

    def __init__(self, config, fields=None, invert=False,
                 scot=None, params=None, 
                 collection=None):
        """

            Arguments 
            config - A RawConfigParser that has alrady read a config file.
            fields The fields to be included from each record.
            invert If true, projects fields that are not in the fields lists.
            scot Interface to scot.
            params Parameters to pass to the query
            collection Which collection to query
        """

        if fields is None:
            raise TopicModelFeatureExtractionException("fields must be " +
                "specified")
        if scot is None:
            raise TopicModelFeatureExtractionException("scot must be " +
                "specified")
        if collection is None:
            raise TopicModelFeatureExtractionException("collection must be" +
                " specified")    
        self._fields = fields
        self._invert = invert
        self._params = params
        self._collection = collection
        self._scot = scot
        self._stop_list = []

        try:
            stop_file = config.get('TopicModelFeatureExtractionSection', 
                                   'stop_words_file')
            self._stop_list = open(stop_file).read().splitlines()
        except Exception as e:
            logging.exception(str(e))
            logging.exception("Error processing stop file, so no stop " +
                              "words list")
    
    def __iter__(self):
       
        records = self._scot.query_collection(self._collection, self._params) 
        if records is not None:
            for record in records:
                doc = self.dict2string(record)
                if len(doc) == 0:
                    continue
                yield [word for word in doc if word not in self._stop_list]


    def __str__(self):
        return "_".join(self._fields)

    '''
        Converts dictionary to a string regardless of if it is a specified 
        field.  
        \param record A dictionary representing an alert.
    '''
    def dict2stringHelper(self, record):
        doc = ""
        for key, value in record.iteritems():
            if isinstance(value, dict):
                doc = (doc + " " + 
                        self.dict2stringHelper(value).encode('utf-8').strip())
            else:
                doc = doc + " " + str(value).encode('utf-8').strip()
        return doc

    def dict2string(self, record):
        doc = ""
        for key, value in record.iteritems():
            if key in self._fields and not self._invert:
                if isinstance(value, dict):
                    doc = " ".join([doc, self.dict2stringHelper(value)])
                else:
                    doc = " ".join([doc, record.get(key, '').encode('utf-8').strip()])
            elif key not in self._fields and self._invert:
                if isinstance(value, dict):
                    doc = " ".join([doc, self.dict2stringHelper(value)])
                else:
                    doc = " ".join([doc, record.get(key, '').encode('utf-8').strip()])
        return doc.lower().split()


class TopicModelFeatureExtraction(AbstractFeatureExtraction):

    """Extract topic model features

    This class extracts topic features from a field in a SCOT alert using
    the gensim library.

    Matthew Hoffman, David M. Blei, Francis Bach. Online Learning
    for Latent Dirichlet Allocation. NIPS (2010).

    build_model(): train the LDA model on the given corpus.
    extract_features(): project a new alert through the model.
    model: instance variable representing the model, in case a developer
        wants to access it directly.

    """

    def __init__(self, corpus=None, config=None):
        """Create an instance of the TopicModelFeatureExtraction class

        arguments:
            corpus: corpus object, should be iterable. See DatabaseCorpus.
            config: A configuration file parser.

        """
        self._corpus = corpus
        self._config = config
        self._dictionary = None
        self.model = None

    def pickle(self, handle):
        """ A custom pickle function since some of the extractors can't
            be pickled directly.

            Arguments:
            handle - A handle to a file that has been opened with "wb"
        """
        pickle.dump(self, handle)

    def unpickle(self, handle):
        other = pickle.load(handle)
        self._corpus     = other._corpus
        self._iterations = other._iterations
        self._num_topics = other._num_topics
        self._batch_size = other._batch_size
        self._dictionary = other._dictionary
        self.model       = other.model


    def __str__(self):
        return "TopicModelFeatureExtraction(corpus=" + str(self._corpus) + ")"

    def build_model(self):
        if self._corpus is None:
            raise TopicModelFeatureExtractionExeption("corpus must be " +
                "specified before building model")

        self._iterations = 100
        self._num_topics = 5
        self._batch_size = 1500

        try:
            self._iterations = int(self._config.get(
                             "TopicModelFeatureExtractionSection","iterations"))
        except Exception as e:
            logging.exception(str(e))
            logging.exception("Couldn't find parameter for iterations.  " +
                              "Using default: " + str(self._iterations))

        try:
            self._num_topics = int(self._config.get(
                         "TopicModelFeatureExtractionSection", "num_topics")) 
        except Exception as e:
            logging.exception(str(e))
            logging.exception("Couldn't find parameter for num_topics.  " +
                              "Using default: " + str(self._num_topics))

        try:
            self._batch_size = int(self._config.get(
                    "TopicModelFeatureExtractionSection", "batch_size"))
        except Exception as e:
            logging.exception(str(e))
            logging.exception("Couldn't find parameter for batch size.  " +
                              "Using default: " + str(self._batch_size))


        self._dictionary = corpora.Dictionary(text for text in self._corpus)
        doc_list = [self._dictionary.doc2bow(text) for text in self._corpus]
        self.model = ldamodel.LdaModel(corpus=doc_list, eval_every=100000,
                                       chunksize=self._batch_size,
                                       num_topics=self._num_topics,
                                       passes=self._iterations)

    def extract_features(self, alert):
        """Project into the model."""
        if self.model is None or self._dictionary is None:
            raise TopicModelFeatureExtractionException("No model built!")
        features = {}
        # Convert to gensim format and project.
        doc = self._corpus.dict2string(alert)
        topic_vector = self.model[self._dictionary.doc2bow(doc)]
        # Output dictionary.
        for i in xrange(len(topic_vector)):
            index = topic_vector[i][0]
            key = str(self._corpus) + '_topic_' + str(index)
            features[key] = topic_vector[i][1]
        return features
