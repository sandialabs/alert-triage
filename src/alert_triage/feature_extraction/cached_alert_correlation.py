"""Extract features from correlated alerts.

This class correlates an alert with other alerts by the entities that
they have in common.  It then uses the correlated alerts to create
features for the alert.  For example, what fraction of promoted alerts
shared at least one entitiy with this alert.

CachedAlertCorrelationException: this is the exception type that should be
raised when exceptions or errors occur.

CachedAlertCorrelation: this class extracts features for an alert by
correlating it with other alerts that share at least one entity with
it.

"""

import copy
import time
import logging
import pickle
from alert_triage.feature_extraction import scot_extractor
from scot_extractor import SCOTExtractorException
from alert_triage.feature_extraction.abstract_feature_extraction import (
    AbstractFeatureExtraction)
    

class CachedAlertCorrelationException(Exception):

    """Use this to raise exceptions within this module."""

    pass


class CachedAlertCorrelation(AbstractFeatureExtraction):
    """Extract features using alerts correlated by entities."""

    def __init__(self, scot=None, labels=None):
        """ Create an instance of the CachedAlertCorrelation class.
        
            Arguments:
            scot -- An instance of the SCOTExtractor class.  This provides
                 access to the SCOT Restful interface.
            labels -- These labels come from the label extraction class 
                (LabelExtraction).  It is a dictionary mapping alert ids to 
                labels (also called weights).   
        """

        self._scot = scot
        self._labels = labels

        # key: alert_id, value: list of entity_id
        self._alert2entity_map = {}
        
        # key: entity_id, value: list of alert_id
        self._entity2alert_map = {}
       
        # This creates a dictionary where the keys are of the form
        # "corr_class_<classname>".  The values are set to zero.
        # This object is later copied to calculate the 
        self._default_counter = {"corr_class_unknown": 0.}

    def pickle(self, handle):
        """ Custom pickle function.

            This object can be pickled directly, so we just pickle self.
        """
        pickle.dump(self, handle)

    def unpickle(self, handle):
        stuff = pickle.load(handle)
        self._scot             = stuff._scot
        self._labels           = stuff._labels
        self._alert2entity_map = stuff._alert2entity_map
        self._entity2alert_map = stuff._entity2alert_map
        self._default_counter  = stuff._default_counter

    def __str__(self):
        return "CachedAlertCorrelation"

    def build_model(self):
        """ Builds the model necessary for extracting features.
            
            This populates two dictionaries, _entity2alert and _alert2entity.
            This allows us to query _alert2entity given an alert_id, and get
            all the related entities.  Then from that list of entities, we 
            get all related alerts.  So functionally, the model is a 
            single alert_id -> many alert_id mapping.   

        """
        logging.info("Building cached alert correlation model")
        if self._scot is None:
            raise CachedAlertCorrelationException("Scot is None, which is" +
                " required to build the model")
        if self._labels is None:    
            raise CachedAlertCorrelationException("No labels provided.")
                

        # TODO creating a set over all the values of the labels dictionary
        # seeems like it might be expensive.  Might be better to just get
        # the set of labels from the LabelExtraction class.
        beg = time.time()
        for value in set(self._labels.itervalues()):
            self._default_counter["corr_class_" + str(value)] = 0.
        logging.info("METRICS: Time create set of labels", 
               time.time() - beg)



        alert_ids = self._labels.keys() # Use this as a param in queries

        numEntities = self._scot.get_record_count_collection(
                        scot_extractor.ENTITY_COLLECTION)

        numAlerts = self._scot.get_record_count_collection(
                        scot_extractor.ALERT_COLLECTION,
                        params = {"id" : alert_ids})

        if numEntities < numAlerts:
            self._build_model_via_entities()
        else:
            self._build_model_via_alerts(alert_ids)

    def _build_model_via_entities(self):
        """ Builds the model by expanding out from entities.
            
            Not sure we need this.  I expect that expanding out from alerts
            will always be cheaper.
        """
        beg = time.time()
        entities = self._scot.query_collection(
                    scot_extractor.ENTITY_COLLECTION)
        logging.info("METRICS: Time to get entities in CachedAlertCorrelation."
                      + "_build_model_via_entities: " + str(time.time() - beg))

        beg = time.time() 
        # Iterate over all the entities 
        for entity in entities:
            entity_id = entity.get("id") 

            # Getting all the alerts associated with the entity
            alerts = self._scot.get_related_items(
                        scot_extractor.ENTITY_COLLECTION,
                        entity_id,
                        scot_extractor.ALERT_COLLECTION)
            self._entity2alert_map[entity_id] = []
            if alerts is not None:
                for alert in alerts:
                    alert_id = alert.get("id")

                    if not alert_id == 0 and alert_id in self._labels:
                        self._entity2alert_map[entity_id].append(alert_id)
                        if self._alert2entity_map.get(alert_id) is None:
                            self._alert2entity_map[alert_id] = []
                        self._alert2entity_map[alert_id].append(entity_id)

            # Getting all the alerts associated with the events associated 
            # with the entity
            events = self._scot.get_related_items(
                        scot_extractor.ENTITY_COLLECTION,
                        entity_id,
                        scot_extractor.EVENT_COLLECTION)
            if events is not None:
                for event in events:
                    event_id = event.get("id")
                    alerts = self._scot.get_related_items(
                                    scot_extractor.EVENT_COLLECTION,
                                    event_id,
                                    scot_extractor.ALERT_COLLECTION)
                    if alerts is not None:
                        for alert in alerts:
                            alert_id = alert.get("id")       
                            if not alert_id == 0 and alert_id in self._labels:
                                self._entity2alert_map[entity_id].append(
                                    alert_id)
                                if self._alert2entity_map.get(alert_id) is None:
                                    self._alert2entity_map[alert_id] = []
                                self._alert2entity_map[alert_id].append(
                                    entity_id)
        logging.info("METRICS: Time to iterate over entities in " +
                     "CachedAlertCorrelation._build_model_via_entities" +
                     str(time.time() - beg))
    
    def _build_model_via_alerts(self, alert_ids):
        beg = time.time()
        numberOfExceptions = 0; #Keeps track of how many exceptions we caught
        numberOfQueries    = 0; #Keeps track of total queries to scot_extractor
        alerts = self._scot.query_collection(
                    scot_extractor.ALERT_COLLECTION,
                    params = {"id":alert_ids})
        logging.info("METRICS: Time to get alerts: " + str(time.time() - beg))
           
        beg = time.time() 
        entity_set = set() 
        numberOfAlerts = len(alerts) #Used for logging metrics
        numberOfEntities = 0 #Used for logging metrics
        for alert in alerts:
            alert_id = alert.get("id")

            entities = None
            try:
                numberOfQueries += 1
                entities = self._scot.get_related_items(
                            scot_extractor.ALERT_COLLECTION,
                            alert_id,
                            scot_extractor.ENTITY_COLLECTION)
            except SCOTExtractorException, e:
                logging.warning("Problems getting entities associated with " + 
                    "alert "+ str(alert_id) + ".  Skipping.")
                logging.warning("SCOTExtractorException: " + str(e))
                numberOfExceptions = numberOfExceptions + 1
            if entities is not None:
                numberOfEntities = numberOfEntities + len(entities)
                for entity_name, entity_dict in entities.iteritems():
                    entity_id = entity_dict.get("id")
                    entity_set.add(entity_id)
                    if entity_id not in self._entity2alert_map:
                        self._entity2alert_map[entity_id] = []
                    self._entity2alert_map[entity_id].append(alert_id)

                    if alert_id not in self._alert2entity_map:
                        self._alert2entity_map[alert_id] = []
                    self._alert2entity_map[alert_id].append(entity_id)
        logging.info("METRICS: Time to iterate over alerts:" +
                    str(time.time() - beg))
        logging.info("METRICS: Total alerts processed (1st loop): " + 
                     str(numberOfAlerts))
        logging.info("METRICS: Total entities processed (1st loop): "
                      + str(numberOfEntities))
        logging.info("METRICS: number of exceptions (1st loop): " +
                      str(numberOfExceptions))
        logging.info("METRICS: number of queries (1st loop): " +
                      str(numberOfQueries))

        event_dictionary = {}
        alert_dictionary = {}

        beg = time.time() 
        numberOfExceptions = 0 #Keeps track of how many exceptions we caught
        numberOfQueries    = 0 #Keeps track of total queries to scot_extractor
        numberOfEvents = 0 #Keeps track of how many events we examine
        numberOfAlerts = 0 #Keeps track of how many alerts we pull from scot
        numberOfCachedAlerts = 0 #how many alerts we pulled from the cache
        logging.info("METRICS: Len of entity set: " + str(len(entity_set)))
        for entity in entity_set:

            try:
                numberOfQueries += 1 
                events = self._scot.get_related_items(
                            scot_extractor.ENTITY_COLLECTION,
                            entity,
                            scot_extractor.EVENT_COLLECTION)
            except SCOTExtractorException, e:
                logging.warning("Problems getting events associated with" +
                                " entity " +  str(entity) + ".  Skipping.")
                logging.warning("SCOTExtractorException: " + str(e))
                numberOfExceptions = numberOfExceptions + 1
            if events is not None:
                numberOfEvents = numberOfEvents + len(events)
                for event in events:
                    event_id = event.get("id")
                    
                    # Check to see if we cached the data
                    if event_id in event_dictionary:
                        alert_id_list = event_dictionary[event_id]
                        numberOfCachedAlerts += len(alert_id_list)
                        alert_list = self._create_alert_list(alert_dictionary,
                                                        alert_id_list)
                        self._process_event_alerts(entity, alert_list)



                    alerts = None
                    try: 
                        numberOfQueries += 1
                        alerts = self._scot.get_related_items(
                                    scot_extractor.EVENT_COLLECTION,
                                    event_id,
                                    scot_extractor.ALERT_COLLECTION)
                    except SCOTExtractorException, e:
                        logging.warning("Problems getting alerts associated " +
                            "with  event " + str(event_id) + ".  Skipping.")
                        logging.warning("SCOTExtractorException: " + str(e))  
                        numberOfExceptions = numberOfExceptions + 1
                    if alerts is not None:
                        # Update metrics
                        numberOfAlerts = numberOfAlerts + len(alerts)

                        # Gather alerts for cache
                        alert_id_list = []
                        for alert in alerts:
                            alert_id = alert.get("id")
                            alert_id_list.append(alert_id)
                            alert_dictionary[alert_id] = alert
                        event_dictionary[event_id] = alert_id_list
                            
                        # Update the entity2alert and alert2entity maps
                        self._process_event_alerts(entity, alerts)

                        #for alert in alerts:
                        #    alert_id = alert.get("id")
                        #    if entity not in self._entity2alert_map:
                        #        self._entity2alert_map[entity] = []
                        #    if not alert_id == 0:
                        #        self._entity2alert_map[entity].append(
                        #          alert_id)
                        #    if alert_id not in self._alert2entity_map:
                        #        self._alert2entity_map[alert_id] = []
                        #    self._alert2entity_map[alert_id].append(
                        #            entity)

        logging.info("METRICS: Time to iterate over entity_set in " +
                    "CachedAlertCorrelation." +
                    "_build_model_via_alerts: " + str(time.time() - beg))
        logging.info("METRICS Total Events processed (2nd loop): "+ 
                    str(numberOfEvents))
        logging.info("METRICS: total alerts processed (2nd loop): "+ 
                    str(numberOfAlerts))
        logging.info("METRICS: alerts cached (2nd loop): "+ 
                    str(numberOfCachedAlerts))
        logging.info("METRICS: number of exceptions (2nd loop): " +
                      str(numberOfExceptions))
        logging.info("METRICS: number of queries (2nd loop): " +
                      str(numberOfQueries))


    def _create_alert_list(self, alert_dictionary, alert_id_list):
        """ Creates a list of alerts from a list of alert ids.

            Arguments:
            alert_dictionary - A dictionary that maps alert ids to alerts.
            alert_id_list - The list of alert ids.
        """
        alert_list = []
        for alert_id in alert_id_list:
            alert_list.append(alert_dictionary[alert_id])
        return alert_list

    def _process_event_alerts(self, entity, alerts):
        """ Takes an entity id and a list of alerts and updates the two maps.

            Arguments:
            entity - The entity id.
            alerts - A list of alerts.

        """
        for alert in alerts:
            alert_id = alert.get("id")
            if entity not in self._entity2alert_map:
                self._entity2alert_map[entity] = []
            if not alert_id == 0:
                self._entity2alert_map[entity].append(
                  alert_id)
            if alert_id not in self._alert2entity_map:
                self._alert2entity_map[alert_id] = []
            self._alert2entity_map[alert_id].append(
                    entity)



    def extract_features(self, alert):
        """Extract correlation features for a given alert."""
        return self._dictionary_match(alert["id"])

    def _dictionary_match(self, alert_id):
        """ Creates a dictionary of correlation results for the given alert. 

            From the single alert id -> many alert id mapping of the model,
            we find all the related alerts for the specified alert_id.
            Then we calculate the fraction of each label found in the list
            of associated alerts.  For example, if alert_id id 1 had related
            alert ids 5, 9, and 10, and the class labels were the following:
            id  label
            5   "promoted"
            9   "closed"
            10  "closed"
            The the following dictionary would be retured:
            { corr_class_promoted: 0.333,
              corr_class_closed: 0.666 }

            Sometimes we are given alerts where the related alerts are not 
            in the original set.  These alerts do not have labels (at least 
            they have not been extracted).  The current strategy for dealing 
            with these is to simply add to the corr_class_unknown set.

            Returns:
                Returns a dictionary where the keys are of the form
                "corr_class_<label>".  The label classes come from the
                LabelExtraction class.  The values represent the 
            
        """
        
        # Make a copy of the counter object (defined in the init function).  
        # Since we need a distinct copy with counts unrelated to other calls 
        # of this function, we make a deep copy.
        class_counter = copy.deepcopy(self._default_counter)

        # Iterate over the entities associatd with this alert
        entities = self._alert2entity_map.get(alert_id, [])
        for entity in entities:
            # Iterate over the alerts associated with the entities
            for alert in self._entity2alert_map[entity]:

                # Increment the counter for the given label by 1
                class_counter["corr_class_" +
                              str(self._labels.get(alert, "unknown"))] += 1.

        total = sum(class_counter.values())
        if total > 0:
            for key in class_counter.keys():
                class_counter[key] = float(class_counter[key]) / float(total)

        return class_counter
