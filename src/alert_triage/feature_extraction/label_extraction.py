"""Extract implicit labels.

This class extracts implicit labels based upon the lifecycle of an
alert. For example, if an alert comes in and is immediately closed, we
mark that alert as a false positive.  We provide a method
(extract_labels) that returns a dictionary that looks like this:

{1: 0, 43: 3, ... }

The keys are the alert ids and the values are the weights of each
alert. (The weights for the various alert types are passed in as an
argument to the LabelExtraction constructor.)  How we derive the
implicit categories is discussed ad nauseum on the wiki, so we refer
the reader there for more information (or look at the code below).

Initially, we made a very clear distinction between implicit labels
extracted from the lifecycle of an alert and the explicit labels
obtained via active learning.  After further discussion, it became
clear that we would obtain labels on the queried alerts by requesting
that an analyst process the alert like they would any other alert.
This is still meaningful because the vast majority of alerts, say 94%,
are never moved from the "open" status, hence we never obtain labels
on them.  The active learning algorithms operate on these "open"
alerts and we essentially nudge the analysts to process them because
of the improvement in rankings that will likely result.

Debug info is controlled through the python logging module.

_IMPLICIT_CLASSES: this module constant defines the various implicit
classes.

_WEIGHTS: this defines the weights/labels given to the various classes
of alerts.

labeled_ids(): this module-level function returns the alert ids for
those alerts that have labels/weights.  

unlabeled_ids(): this module-level function returns the alert ids for
those alerts that don't have labels/weights.

LabelExtractionException: this is the exception type that should be
raised when exceptions or errors occur.

LabelExtraction: this has a method, extract_labels, that will return
the labels for all the alerts.

"""

import collections
import random
import time
import logging
import sys
from alert_triage.feature_extraction import scot_extractor

_WEIGHTS = {"false_positive": 0,
            "open_not_viewed": 1,
            "open_viewed": 2,
            "revisit": 3,
            "promoted_false_positive": 4,
            "promoted": 5,
            "incident": 6
            }

_IMPLICIT_CLASSES = ["false_positive", "open_not_viewed", "open_viewed",
                     "revisit", "promoted_false_positive", "promoted",
                     "incident"]


def labeled_ids(alert_labels):
    """Return the alert ids for all the labeled alerts.
    
        This takes a dictionary of alert_labels that was produced by the
        label_extraction class and returns the ids of alerts that are 
        labeled.

        Args:
            alert_labels: A dictionary where the keys are alert ids and 
            the values are the labels.

        Returns:
            Returns a list of alert ids.
    
    """
    return [key for key, value in alert_labels.iteritems()
            # If the label for an alert is "ignore", it means it is
            # not part of the labeled or unlabeled sets.
            if value is not None and value != "ignore"]

def unlabeled_ids(alert_labels):
    """Return the alert ids for all the unlabeled alerts.
    
        This takes a dictionary of alert_labels that was produced by the
        label_extraction class and returns the ids of alerts that are 
        not labeled.

        Args:
            alert_labels: A dictionary where the keys are alert ids and 
            the values are the labels.
    
        Returns:
            Returns a list of alert ids.
    """
    return [key for key, value in alert_labels.iteritems()
            if value is None]

class LabelExtractionException(Exception):

    """Exception type for the LabelExtraction class."""

    pass


class LabelExtraction(object):

    """Extract implicit labels.

    This class extracts implicit labels (actually, weights) for alerts
    based on the lifecycle of an alert.

    alert_labels: this instance variable allows access to the alert
    ids with their corresponding labels/weights.

    label_counts: this instance variable tells you how many alert
    instances there are for each label/weight.

    extract_labels(): this is the workhorse method of the class. It
    calls several private methods to label the various alert types.

    """

    def __init__(self, scot, limit=0, 
                 begin=None, end=None,
                 alert_ids=None, 
                 weights=_WEIGHTS):
        """Create an instance of the LabelExtraction class.

        The constraints (limit, begin, end, alert_ids) only apply to the
        alert collection.  

        arguments:
            scot: an interface to scot
            limit: how many alerts to extract labels for.  A limit of
            0 means extract labels for all the alerts.  This doesn't affect
            queries against other collections (e.g. alertgroup, events).
            begin: only extract labels for alerts created after this
            date (in seconds since epoch).  Must be specified with end.
            end: only extract labels for alerts created before this date
            (in seconds since epoch).  
            alert_ids: if specified, a list with alerts are to be labeled. 
            weights: a dictionary mapping classes to weights

        """
        self._scot = scot
        self._limit = limit

        if begin is not None: 
            if end is None:
                end = int(time.time())

        if end is not None:
            if begin is None:
                raise LabelExtractionException("If end is specified, begin " +
                                "must also be specified")
        self._begin = begin
        self._end = end
        self._alert_ids=alert_ids
        self._weights = weights
        self.alert_labels = {}
        self.label_counts = collections.Counter()

    def extract_labels(self):
        """Extract the implicit labels."""
        logging.info("Entering extract_labels")
        time1 = time.time()
        self._find_false_positive()
        logging.info("Time for _find_false_positive:" + str(time.time() -time1))
        time1 = time.time()
        self._find_open_not_viewed()
        logging.info("Time for _find_open_not_viewed:"+ str(time.time() -time1))
        time1 = time.time()
        self._find_open_viewed()
        logging.info("Time for _find_open_viewed:"+ str(time.time() - time1))
        time1 = time.time()
        self._find_revisit()
        logging.info("Time for _find_revisit:"+ str(time.time() - time1))
        time1 = time.time()
        self._find_promoted_false_positive()
        logging.info("Time for _find_promoted_false_positive:"
            + str(time.time() - time1))
        time1 = time.time()
        self._find_incident()
        logging.info("Time for _find_incident:" + str(time.time() - time1))
        time1 = time.time()
        # We must get the promoted false positives and incidents
        # before running this method since promoted false positive and
        # indicent alerts are not part of the promoted set.
        self._find_promoted()
        if self._limit != 0 and len(self.alert_labels) > self._limit:
            alert_ids = [key for key in self.alert_labels.iterkeys()]
            del_ids = random.sample(alert_ids, len(self.alert_labels) 
                                    - self._limit)
            for key in del_ids:
                del self.alert_labels[key]
        logging.info("Returning alert labels")
        return self.alert_labels

    def _add_common_alert_params(self, params):
        """ Adds common parameters for alert queries.

            Adds to a parameter dictionary options that were set at 
            construction.  Namely:
            1) limit
            2) created boundaries
            3) a list of specific alert ids.
            The result is a intersection of all of those options along
            with any label-specific parameters.

            Args:
                params: An existing dictionary.  The common parameters
                will be added to this dictionary.

            Returns:
                A dictionary with the common parameters added.
        """

        # Don't want to pass a limit=0 or else the server will not
        # respond. 
        if self._limit != 0:
            params["limit"] = self._limit
       
        if self._begin is not None:
            if self._end is not None:
                params["created"] = [self._begin, self._end]

        if self._alert_ids is not None:
            params["id"] = self._alert_ids

        return params

    def _find_false_positive(self):
        """ Extract labels for all the false positive alerts.

            Alerts that are closed are considered false positives.  All
            the closed alerts found, subject to the constraints specified
            (e.g. time boundaries, limit, specific alert ids), are added
            class member alert_labels dictionary and a weight 
            is assigned (i.e. the weight assigned to the category 
            "false_positive").

            Args:
                No arguments except self.

            Returns:
                Does not return anything.  Has the side-effect of modifying
                the alert_labels class member variable.
        """
        
        params = {}
        params["status"] = "closed"

        params = self._add_common_alert_params(params)        
        
        logging.info("params " + str(params))

        # These should be all the closed alerts that satisfy the other
        # constraints. 
        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION, 
                                             params)
        if alerts is not None: logging.info("len(alerts) " + str(len(alerts)))
        else: logging.info("alerts is None")

        # Adding the found alerts to the alert_labels dictionary
        if alerts is not None:
            for alert in alerts:
                weight = self._weights.get("false_positive", None)
                self.alert_labels[alert.get("id")] = weight
                if weight is not None and weight != "ignore":
                    self.label_counts[weight] += 1


    def _find_open_not_viewed(self):
        """ Extract labels for all open alerts that haven't been viewed.
      
           Args:
                No arguments except self.

           Returns:
                Does not return anything.  Has the side-effect of modifying
                the alert_labels class member variable.

        """

        # We first have to query alertgroup to get all alertgroups that
        # have views=0.  Then, once we have those we iterate through
        # those alertgroups, iterate through the alerts for each alertgroup,
        # and find all alerts that have status=open.
        alertgroupParams = {}
        alertgroupParams["views"] = 0
        logging.info("alertgroupParams " + str(alertgroupParams))


        alertgroups = self._scot.query_collection(
                        scot_extractor.ALERT_GROUP_COLLECTION, 
                        alertgroupParams) 
      
        # First iterate through the alert groups that don't have any views 
        if alertgroups is not None: 
            for alertgroup in alertgroups:
                alertgroup_id = alertgroup["id"]
                if alertgroup_id is not None:
                    # Now for the alert group, we get the related alerts
                    alerts = self._scot.get_related_items(
                        scot_extractor.ALERT_GROUP_COLLECTION,
                        alertgroup_id,
                        scot_extractor.ALERT_COLLECTION)

                    # For the alerts in the alert group, add the open_not_viewed
                    # weight for each of the alerts.
                    if alerts is not None:
                        for alert in alerts:

                            # Have to check if the alert_id is in the list if 
                            # alert_ids has been specified.
                            if ((self._alert_ids is not None and 
                                alert["id"] in self._alert_ids) or 
                                self._alert_ids is None):
                                alertparams = {}
                                alertparams = self._add_common_alert_params(
                                                alertparams)
                                alertparams["id"] = alert["id"]
                                check = self._scot.query_collection(
                                    scot_extractor.ALERT_COLLECTION,
                                    alertparams)

                                if check is not None:
                                    weight = self._weights.get(
                                        "open_not_viewed", None)
                                    self.alert_labels[alert.get("id")] = weight
                                    if weight is not None and weight !="ignore":
                                        self.label_counts[weight] += 1


    def _find_open_viewed(self):
        """ Finds alerts that have been opened and viewed.

            Because this query can be expensive, we first find the number
            of alertgroups that have been viewed and the number of alerts.
            Whichever is a smaller set, we perform the query from that smaller
            set.
        """

        # Set the parameters for the alertgroup query.  Note that we don't
        # include the limit in this query.  The limit applies to the alerts 
        # only.
        alertGroupParams = {}
        alertGroupParams["views"] = "x>0"
   
        # Set the parameters for the alert query. 
        alertParams = {}
        #Adds limit, created, alert_ids constraints
        alertParams = self._add_common_alert_params(alertParams) 
        alertParams['status'] = "open"

        numAlertGroups = self._scot.get_record_count_collection(
                             scot_extractor.ALERT_GROUP_COLLECTION,
                             alertGroupParams)

        numAlerts = self._scot.get_record_count_collection(
                        scot_extractor.ALERT_COLLECTION, alertParams)

        logging.info("numAlertGroups: " + str(numAlertGroups))
        logging.info("numAlerts: " + str(numAlerts))

        if numAlertGroups < numAlerts:
            self._find_open_viewed_via_alertgroups(alertGroupParams)
        else:
            self._find_open_viewed_via_alerts(alertParams)
       
       
    def _find_open_viewed_via_alertgroups(self, alertGroupParams):
        """ Finds the open viewed alerts by searching through the alertgroups
            first.

            This is called when the number of viewed alert groups is smaller
            than the number of alerts.

            This doesn't return anything but instead modifies the class member
            variables alert_labels and label_counts.

            Arguments:
            alertGroupParams - The parameters to pass to scot to collect the
                relevant alertgroups.
        """  
        alertgroups = self._scot.query_collection(
                        scot_extractor.ALERT_GROUP_COLLECTION,
                        alertGroupParams)


        # We first have to query alertgroup to get all alert groups that
        # have views>0.  Then once we have those we iterate
        # through the alertgoups, iterate through the alerts for each 
        # alertgroup, and find all alerts that have status=open.
        if alertgroups is not None:
            for alertgroup in alertgroups:
                alertgroup_id = alertgroup["id"]
                if alertgroup_id is not None:
                    alerts = self._scot.get_related_items(
                        scot_extractor.ALERT_GROUP_COLLECTION,
                        alertgroup_id,
                        scot_extractor.ALERT_COLLECTION)
                    if alerts is not None:
                        for alert in alerts:
                            if ((self._alert_ids is not None and 
                                alert["id"] in self._alert_ids) or 
                                self._alert_ids is None):
                                if alert["status"] == "open":
                                    # Maybe TODO: Not checking against limit
                                    # or created time
                                    weight = self._weights.get("open_viewed", 
                                                               None)
                                    self.alert_labels[alert.get("id")] = weight
                                    if weight is not None and weight !="ignore":
                                        self.label_counts[weight] += 1

    def _find_open_viewed_via_alerts(self, params):
        """ Finds the open viewed alerts by searching through the alerts first.

            This is called when the number of alerts is smaller than the number
            of alertgroups.

            This doesn't return anything but instead modifies the class member
            variables alert_labels and label_counts.

            Arguments: 
            params - The parameters for the query against the alert collection.
        """

        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                    params)

        if alerts is not None:
            for alert in alerts:
                alertgroup_id = alert["alertgroup"]
                alertgroupParams = {"id": alertgroup_id}

                # Get the alertgroup that this alert is part of.
                alertgroup = self._scot.query_collection(
                                scot_extractor.ALERT_GROUP_COLLECTION,
                                alertgroupParams)
                print "alertgroup ", alertgroup_id, alertgroup[0].keys()
                print "_id", alertgroup[0]["_id"]
                print "alertgroup", alertgroup[0]
                if alertgroup is None or len(alertgroup) != 1:
                    raise LabelExtractionException("Expected to find one" +
                        " alertgroup with id " + str(alertgroup_id) + " but" +
                        " instead found none or more than one")
                if "views" not in alertgroup[0]:
                    raise LabelExtractionException("alertgroup with id " +
                        str(alertgroup_id) + " does not have views field") 
                if alertgroup[0]["views"] > 0:  
                    weight = self._weights.get("open_viewed", None)
                    self.alert_labels[alert.get("id")] = weight
                    if weight is not None and weight != "ignore":
                        self.label_counts[weight] += 1

                


    def _find_revisit(self):
        # Extract labels for all the alerts marked to revisit.
        params = {"status": "revisit"}

        params = self._add_common_alert_params(params)        

        logging.info("params " + str(params))

        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             params)

        if alerts is not None:
            for alert in alerts:
                weight = self._weights.get("revisit", None)
                self.alert_labels[alert.get("id")] = weight
                if weight is not None and weight != "ignore":
                    self.label_counts[weight] += 1

    def _find_promoted_false_positive(self):
        # Extract labels for all the promoted alerts that were later
        # marked as false positives.
        # Don't know if we can query the tags, so we'll get everything
        # and do the filter ourselves.

        params = {}
        logging.info("params " + str(params))

        events = self._scot.query_collection(scot_extractor.EVENT_COLLECTION, 
                                             params)
                              
        if events is not None:               
            for event in events:
                tags = event.get("tag")
                for tag in tags:
                    if (tag == "false_positive" or 
                       tag == "false positive" or 
                       tag == "falsepositive"):
                           
                        event_id = event.get("id")
                        alerts = self._scot.get_related_items(
                                    scot_extractor.EVENT_COLLECTION,
                                    event_id,
                                    scot_extractor.ALERT_COLLECTION)
                        if alerts is not None:
                            for alert in alerts:
                                # Check to see if alert id is in the alert list
                                # if it exists.
                                if ((self._alert_ids is not None and 
                                    alert["id"] in self._alert_ids) or 
                                    self._alert_ids is None):
                                    # Do another query, but this time with the
                                    # constraints specified.  

                                    alertParams = {}
                                    alertParams = self._add_common_alert_params(
                                                    alertParams)
                                    alertParams["id"] = alert["id"]
                                    singleAlert = self._scot.query_collection(
                                        scot_extractor.ALERT_COLLECTION,
                                        alertParams)

                                    if singleAlert is not None: 
                                        # passed contraints, so add
                                        self.alert_labels[alert["id"]] = weight
                                        if (weight is not None and 
                                            weight != "ignore"):
                                            self.label_counts[weight] += 1
                            

    def _find_incident(self):
        """ Extract labels for all the alerts associated with incidents.
        """

        eventParams = {}
        logging.info("eventParams " + str(eventParams))

        alertParams = {}
        alertParams = self._add_common_alert_params(alertParams)
        logging.info("alertParams " + str(alertParams))

        incidentParams = {}
        logging.info("incidentParams " + str(incidentParams))

        numEvents = self._scot.get_record_count_collection(
                            scot_extractor.EVENT_COLLECTION,
                            eventParams)
        
        numAlerts = self._scot.get_record_count_collection(
                            scot_extractor.ALERT_COLLECTION,
                            alertParams)
        
        numIncidents = self._scot.get_record_count_collection(
                            scot_extractor.INCIDENT_COLLECTION,
                            incidentParams)

        logging.info("Number of events: " + str(numEvents))
        logging.info("Number of alerts: " + str(numAlerts))
        logging.info("Number of incidents: " + str(numIncidents))
       
        # If the number of alerts is less than events and incidents, expand from
        # alerts first.
        if (numAlerts <= numEvents) and (numAlerts <= numIncidents):
            self._find_incident_via_alerts(alertParams)
        # If the number of incidents is less than alerts and events, expand from
        # incidents first. 
        elif (numIncidents <= numAlerts) and (numIncidents <= numEvents):
            self._find_incident_via_incidents(incidentParams)    
        # The number of events is less than alerts and incidents, so expand from
        # events.
        else:
            self._find_incident_via_events(eventParams)    

    def _find_incident_via_incidents(self, incidentParams):
        """ Finds alerts that are related to incicents by expanding out
            from incidents first

            Does not return anything but modifies class member variables
            alert_labels and label_counts.

            Arguments:
            incidentParams - The parameters to get the set of incidents from
            scot that we expand from to find related alerts.
        """
        logging.debug("Entering _find_incident_via_incidents")
        logging.info("incidentParams " + str(incidentParams))
        incidents = self._scot.query_collection(
                        scot_extractor.INCIDENT_COLLECTION,
                        incidentParams)
        logging.info("Number of incidents: " + str(len(incidents)))

        if incidents is not None:
            for incident in incidents:
                incident_id = incident["id"]
                logging.debug("incident id " + str(incident_id))
                events = self._scot.get_related_items(
                            scot_extractor.INCIDENT_COLLECTION,
                            incident_id,
                            scot_extractor.EVENT_COLLECTION)
                if events is not None:
                    for event in events:
                        event_id = event["id"]
                        alerts = self._scot.get_related_items(
                            scot_extractor.EVENT_COLLECTION,
                            event_id,
                            scot_extractor.ALERT_COLLECTION)
                        if alerts is not None:
                            for alert in alerts:
                                alert_id = alert["id"]
                                weight = self._weights.get("incident", None)
                                self.alert_labels[alert.get("id")] = weight
                                if weight is not None and weight != "ignore":
                                    self.label_counts[weight] += 1
        logging.debug("Exiting _find_incident_via_incidents")


    def _find_incident_via_alerts(self, alertParams):
        """ Finds alerts that are related to incidents by expanding out
            from alerts first

            Arguments:
            alertParams - The parameters to get the set of alerts from scot that
                we then expand from and see if they are related to incidents.
        """
        logging.debug("Entering _find_incident_via_alerts")
        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             alertParams)
        logging.info("Number of alerts: " + str(len(alerts)))

        if alerts is not None:
            for alert in alerts:
                alert_id = alert["id"]
                events = self._scot.get_related_items(
                            scot_extractor.ALERT_COLLECTION,
                            alert_id,
                            scot_extractor.EVENT_COLLECTION)
                if events is not None:
                    for event in events:
                        event_id = event["id"]
                        incidents = self._scot.get_related_items(
                                    scot_extractor.EVENT_COLLECTION,
                                    event_id,
                                    scot_extractor.INCIDENT_COLLECTION)
                        if incidents is not None:
                            #the alert is associated with an incident so add it
                            weight = self._weights.get("incident", None)
                            self.alert_labels[alert.get("id")] = weight
                            if weight is not None and weight != "ignore":
                                self.label_counts[weight] += 1
        logging.debug("Exiting _find_incident_via_alerts")


    def _find_incident_via_events(self, eventParams):   
        """ Finds alerts that are related to incidents by expanding out 
            from events first.

            Arguments:
            eventParams - The parameters to get the events from scot.

        """     
        logging.debug("Entering _find_incident_via_events")
        events = self._scot.query_collection(scot_extractor.EVENT_COLLECTION,
                                             eventParams)
        logging.info("number of events" + str(len(events)))

        if events is not None:
            for event in events:
                event_id = event.get("id")
                # We go through all the events and get the ids for 
                # incidents.
                incidents = self._scot.get_related_items(
                                        scot_extractor.EVENT_COLLECTION,
                                        event_id,
                                        scot_extractor.INCIDENT_COLLECTION) 
                if incidents is not None:
                    # For each incident we find related alerts.  Those
                    # alerts get the label of "incident"
                    alerts = self._scot.get_related_items(
                                        scot_extractor.EVENT_COLLECTION,
                                        event_id,
                                        scot_extractor.ALERT_COLLECTION) 
                    if alerts is not None:
                        for alert in alerts:
                            if ((self._alert_ids is not None and 
                                alert["id"] in self._alert_ids) or 
                                self._alert_ids is None):
                                weight = self._weights.get("incident", None)
                                self.alert_labels[alert.get("id")] = weight
                                if weight is not None and weight != "ignore":
                                    self.label_counts[weight] += 1
        logging.debug("Exiting _find_incident_via_events")


    def _find_promoted(self):
        # Find labels for all the promoted alerts.
        params = {"status": "promoted"}
        params = self._add_common_alert_params(params)

        logging.info("params " + str(params))

        alerts = self._scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                             params)

        if alerts is not None:
            for alert in alerts:
                # If this has already been added as a promoted false
                # positive or an incident, don't add it again.
                if not alert["id"] in self.alert_labels:
                    weight = self._weights.get("promoted", None)
                    self.alert_labels[alert["id"]] = weight
                    if weight is not None and weight != "ignore":
                        self.label_counts[weight] += 1
