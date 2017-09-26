"""Extract features from the extracted entities collection.

This class extracts features for alert instances using the extracted
entities collection in the scot database.  The extracted entities
collection is built by the scot backend and provides connections
between alerts based on entities found within alerts.  Examples of
entities are ip addresses, email addresses, and domain names.

To run, an object of the class is instantiated, extract features
is called, and then the public features variable is accessed.

ExtractedEntitiesException: this is the exception type that should be
raised when exceptions or errors occur. This isn't currently used.

"""

import ConfigParser
import pygeoip
import pickle
import re
import socket
import time
import logging
import alert_triage
from alert_triage.feature_extraction import scot_extractor 
from alert_triage.feature_extraction.abstract_feature_extraction import (
    AbstractFeatureExtraction)

class ExtractedEntitiesException(Exception):

    """Exception type for the ExtractedEntities Class"""

    pass


class ExtractedEntities(AbstractFeatureExtraction):

    """Extract extracted entities features

    This class extracts features from the extracted entities collection for
    a single alert.

    extract_features(): This method does all the work for the class, and
    calls the code that extracts features from ip addresses, domain
    names and email addresses.  After calling, the features class
    variable contains the feature set.

    features: This instance variable contains a dictionary of the
    extracted features.  The key of the dictionary is the alert_id

    alert: The alert to extract features from

    """

    def __init__(self, scot=None, config=None):
        """Create an instance of the Batch Pipeline class

        Arguments:
        scot - A SCOTExtractor object.
        config - A RawConfigParser that has already read a config file.  The
            geoip files are found via the config object.
        """

        self._scot = scot

        self._geoISPFile = None
        self._geoCityFile = None
        self._geo_ISP = None 
        self._geo_city = None 
        self._limit = 0
        if config is not None:
            logging.info("Config file found.")
            try:
                self._geoISPFile =config.get('ExtractedEntitiesSection', 
                                             'geo_isp_file')
                self._geoCityFile=config.get('ExtractedEntitiesSection',
                                             'geo_city_file')
                self._geo_ISP = pygeoip.GeoIP(self._geoISPFile)
                self._geo_city = pygeoip.GeoIP(self._geoCityFile)
            except Exception as e:
                logging.exception(str(e))
                logging.exception("Problem setting up geo ip.  "+ 
                                  "No geo ip features")
                self._geo_ISP = None 
                self._geo_city = None
            try:
                self._limit = config.get('ExtractedEntitiesSection',
                                         'limit') 
            except Exception as e:
                logging.exception(str(e))
                logging.exception("Problem getting limit.  Using 0.")
        else:
            logging.info("Config is none, so no geo ip features and limit is" +
                         " zero.")

    def pickle(self, handle):
        """ Custom pickle function.

            Since pygeoip creates a lock object, it can't be pickled.  Thus we
            have pickle with this custom function.

            Arguments:
            handle - The handle to a pickle file that has already been opened
                     with "wb"
        """
        if (self._scot is None):
            raise ExtractedEntitiesException("Attempted to pickle Extracted" +
                "Entities object but scot is None.   A SCOTExtractor object "+
                "is required") 
        pickle.dump(self._scot, handle)
        pickle.dump(self._limit, handle)
        pickle.dump(self._geoISPFile, handle)
        pickle.dump(self._geoCityFile, handle)

    def unpickle(self, handle):
        """ Custom pickle load function.
            
            Since pygeoip creates a lock object, it can't be pickled.  Thus we
            have to load the object with this custom function.
            
            Arguments:
            handle - The handle to a pickle file that has already been opened
                     with "rb"
        """
        self._scot = pickle.load(handle)
        self._limit = pickle.load(handle)

        # The geoip stuff wasn't pickled because it has a lock object that
        # can't be pickled.  We get the file names from the pickle and
        # reinstantiate
        self._geoISPFile = pickle.load(handle)
        self._geoCityFile = pickle.load(handle)
        self._geo_ISP = pygeoip.GeoIP(self._geoISPFile)
        self._geo_city = pygeoip.GeoIP(self._geoCityFile)

    def extract_features(self, alert):
        """ Extracts features for the given alert.

            Arguments:
            alert - This parameter is expected to be a dictionary that 
                         represents an alert.

            Return:
            Returns a dictionary where the key is a feature label and the 
            values are the feature values.

        """ 

        if self._scot is None:
            raise ExtractedEntitiesException("No scot object.  SCOT is " +
                "required to extract any features.")
        if self._geo_ISP is None:
            logging.debug("No geo_ISP object.  Features requiring it" +
                          " will not be produced")
        if self._geoCityFile is None:
            logging.debug("No geo_city object.  Features requiring it" +
                          " will not be produced")
        alert_id = alert.get("id")
        params = {"limit":self._limit}
        
        entities = self._scot.get_related_items(scot_extractor.ALERT_COLLECTION,
                                               alert_id,
                                               scot_extractor.ENTITY_COLLECTION,
                                               params)

        ips = self._scot.get_entity_type_from_dictionary(entities, 
                scot_extractor.IP_ENTITY) 
        domains = self._scot.get_entity_type_from_dictionary(entities, 
                scot_extractor.DOMAIN_ENTITY)
        emails = self._scot.get_entity_type_from_dictionary(entities, 
                scot_extractor.EMAIL_ENTITY)
        files = self._scot.get_entity_type_from_dictionary(entities, 
                scot_extractor.FILE_ENTITY)


        features = {}
        if alert_id:
            features.update(self._get_ip_features(ips))
            features.update(self._get_domain_features(domains))
            features.update(self._get_email_features(emails))
            features.update(self._get_file_features(files))
        return features

    def build_model(self):
        """ No model with this feature extractor, but implementing abstract 
            method of AbstractFeatureExtraction.
        """
        pass

    def _get_simple_count_features(self, entities, label):
        """ Gets common features related to counts
         
            Gets some features based on counts of alerts, entries, events, and 
            incidents from a list of entities that have been filtered to be of 
            a single type, either "email" or "file" (domains and ips are handled
            with different methods).  The list of entities are all associated
            with a particular alert.  For example, features "total_email_events"
            and "average_email_events" are created.  "total_email_events" is the
            total number of events associated with emails coming from a single
            alert.  "average_email_events" is the average number of events 
            associated with each email associated with a particular alert.

            Arguments:
            entities This is a dictionary of entities that have been
                            previously filtered to be either only emails or
                            files.
            label Either "email" or "file".  The label is used in the name
                         of the feature.

            Return: 
            Returns a dictionary of features where the key is the name
                of the feature and the value is the feature, a float.

                            
        """

        features = {}
        params = {"limit": self._limit}
        count = len(entities)
        total_events = 0
        total_entries = 0
        total_alerts = 0
        total_incidents = 0
        for entity in entities:
            entity_id = entities[entity].get("id")
            entries_list =  self._scot.get_related_items(
                                             scot_extractor.ENTITY_COLLECTION,
                                             entity_id,
                                             scot_extractor.ENTRY_COLLECTION,
                                             params) 
            alert_list = self._scot.get_related_items(
                                             scot_extractor.ENTITY_COLLECTION,
                                             entity_id,
                                             scot_extractor.ALERT_COLLECTION,
                                             params) 
            event_list = self._scot.get_related_items(
                                             scot_extractor.ENTITY_COLLECTION,
                                             entity_id,
                                             scot_extractor.EVENT_COLLECTION,
                                             params) 
            incident_list = self._scot.get_related_items(
                                             scot_extractor.ENTITY_COLLECTION,
                                             entity_id,
                                             scot_extractor.INCIDENT_COLLECTION,
                                             params) 
            if event_list:
                total_events += len(event_list)
            if entries_list:
                total_entries += len(entries_list)
            if alert_list:
                total_alerts += len(alert_list)
            if incident_list:
                total_incidents += len(incident_list)
        if count:
            count = float(count)
            features[label + '_count'] = count
            if total_events:
                features['total_' + label + '_events'] = float(total_events)
                features['average_' + label + '_events'] =  (
                    float(total_events) / count)
            if total_entries:
                features['total_' + label + '_entries'] = float(total_entries)
                features['average_' + label + '_entries'] =  (
                    float(total_entries) / count)
            if total_alerts:
                features['total_' + label + '_alerts'] = float(total_alerts)
                features['average_email_alerts'] = float(total_alerts) / count
            if total_incidents:
                features['total_' + label + '_incidents'] = (
                    float(total_incidents))
                features['average_' + label + '_incidents'] = ( 
                    float(total_incidents) / count)
        return features


    '''
        Right now the only features extracted for emails are the simple
        count features created in _get_simple_count_fetures.
        \param emails A dictionary of emails (from entity collection) 
                      that are associated with a single alert.

    '''
    def _get_email_features(self, emails):
        """Extract features related to email addresses"""
        return self._get_simple_count_features(emails, "email")


    '''
        Right now the only features extracted for files are the simple
        count features created in _get_simple_count_features.
        \param files A dictrionary of files (from entity collection)
                     that are associated wtih a single alert.
    '''
    def _get_file_features(self, files):
        """Extract features related to files"""
        return self._get_simple_count_features(files, "file")

    ''' 
        Gets features for a dictionary of domains that are associated
        with a single alert.  This method queries to find out if 
        the domains are blocked.  If they are, it performs a full query
        to get more information on the domain. 
        \param domains A dictionary of domains (from entity collection)
                       that are associated with a particular alert.    
    '''
    def _get_domain_features(self, domains):
        '''Extract domain features and return list'''

        features = {}
        return features

    '''
        This method converts both IPv4 and IPv6 from the
        ddd.ddd.ddd.ddd and x:x:x:x:x:x:x:x text formats into
        binary form.  This makes it easier to compare. 
        \param ipaddr An ip address in text form.  
    '''
    def _checkipaddr(self, ipaddr):
	'''Check ip address, more thorough than regex'''
    	try:
            return socket.inet_aton(ipaddr)
    	except socket.error:
	    try:
                return socket.inet_pton(socket.AF_INET6, ip)
	    except:
		return False

    def _get_ip_features(self, ips):
        """ Extract ip features and return dictionary of features
        """
        features = {}

        if self._geo_city is not None and self._geo_ISP is not None:
            # Get features from eo city
            for ipaddr in ips:
                ip_match = self._checkipaddr(ipaddr)
                if ipaddr and ip_match is not False:
                        _geo_city_record = self._geo_city.record_by_addr(ipaddr)
                        _org = self._geo_ISP.org_by_addr(ipaddr)
                        if _geo_city_record:
                            features["country_"
                                +_geo_city_record.get("country_code")] = 1
                        if _org:
                            _org = re.sub("[ !\.\-,\/\']", '_', _org).lower()
                            features["organization_" + _org] = 1
        else:
            logging.debug("No geo ip objects defined, so some ip features " +
                          "not generated")
                               
        return features
