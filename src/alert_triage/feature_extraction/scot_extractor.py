
import requests
from requests import ConnectionError
from simplejson import JSONDecodeError
import json
import time
import numbers
import logging


ALERT_COLLECTION  = "alert"
ENTITY_COLLECTION = "entity"
EVENT_COLLECTION = "event"
ALERT_GROUP_COLLECTION = "alertgroup"
INCIDENT_COLLECTION = "incident"
ENTRY_COLLECTION = "entry"

EMAIL_ENTITY = "email"
FILE_ENTITY = "file"
IP_ENTITY = "ipaddr"
DOMAIN_ENTITY = "domain"

ID_LIST_LENGTH_LIMIT = 200

class SCOTExtractorException(Exception):

    """Exception type for the SCOTExtractor class."""

    pass



class SCOTExtractor:

    def __init__(self, base_url):
        self.session = requests.Session()
        self.base_url = base_url
        self.headers = {'X-Requested-With': 'XMLHttpRequest'}
        
        # The increment is so that we don't create a query that is too
        # large and the server gets bogged down.  We instead send
        # multiple queries of size increment or less and aggregate the results.
        self.increment = 1000


    def get_record_count_long(self, url, params):
        """ Obtains the number records when the uri is too long.

            The only case this method currently supports is when the parameter
            id is a list that is too long and would cause 414 too long uri
            errors.  It breaks up the id list into multiple chunks and adds
            up the results of the chunks to get the total record count
            of the entire query.

            Args:
                url The full url to SCOT including the collection information.
                
                params The parameters being added to the URL and passed to SCOT. 

            Returns:
                Returns the total number of items that would be returned by a 
                query to SCOT with the specified url and parameters.
        """
        if params is None:
            raise SCOTExtractorException("Called get_record_count_long without"+
              " params defined")
        if params is not None and 'id' not in params:
            raise SCOTExtractorException("Called get_record_count_long without"+
              " id list defined")

        length = 0
        alert_ids = params['id']
        i = 0
        tmpList = []
        for alert_id in alert_ids:
            tmpList.append(int(alert_id))
            i = i + 1
            if i >= ID_LIST_LENGTH_LIMIT:
                smallParams = dict(params)
                smallParams['id'] = tmpList
                length = length + self.get_record_count(url, smallParams)
                i = 0
                tmpList = []

        return length

    def get_record_count_collection(self, collection, params=None):
        """ Forms url from specified collection and gets the count of the query.
        """
        url = self.base_url + "/" + collection
        return self.get_record_count(url, params)

    def get_record_count(self, url, params=None):
        """ Gets the total record count for the specified url and parameters.

            Args:
                url The full url to SCOT including the collection information.
                
                params The parameters being added to the URL and passed to SCOT. 

            Returns:
                Returns the total number of items that would be returned by a 
                query to SCOT with the specified url and parameters.
        """
        logging.info("Entered get_record_count")
        logging.info("BaseUrl: " + url)
        logging.info("Parameters: " + str(params))

        if params == None:
            params = {}

        # This is to support long id lists.  
        if ('id' in params and isinstance(params['id'], list) and 
            len(params['id']) > ID_LIST_LENGTH_LIMIT ):
            logging.info("The list of ids is too long; spliting into multiple" +
                     " queries: " + str(len(params['id']))+ " vs " + 
                     str(ID_LIST_LENGTH_LIMIT))
            return self.get_record_count_long(url, params)


        response = None

        # Occasionally there is a connection error.  We try max_num_tries
        # times before giving up.
        max_num_tries = 10
        worked = False
        try_num = 0
        total = 0

        while not worked and try_num < max_num_tries:
            worked = True
            try:
                response = self.session.get(url, verify=False, params=params,
                                        headers=self.headers)
            except ConnectionError:
                logging.warn("Connection Exception in get_record_count")
                worked = False
                if response is not None:
                    logging.warn("Request url " + response.url)
                else:
                    logging.warn("Response was None")
                    logging.warn("Parameters used: " + str(params))

            if response is None:
                logging.error("Response was none")
                worked = False
            if response is not None and response.status_code != 200:
                logging.error("Error getting record count, status code " + 
                              str(response.status_code)) 
                logging.error("Error with request url " + response.url)
                worked = False 
           
            if worked: 
                try:
                    total = response.json()['totalRecordCount']
                except JSONDecodeError as e:
                   logging.exception(str(e))
                   logging.error("Error decoding json, trying again.")
                   worked = False 
            try_num = try_num + 1
            if not worked:
                logging.warning("Trying " + str((max_num_tries - try_num -1)) +
                             " more times")

        if try_num >= max_num_tries:
            raise SCOTExtractorException("Couldn't connect after a max number" +
                    " of tries: ", max_num_tries) 

        logging.info("Total reported by server: " + str(total))

        #totalRecordCount still reports the total, even if a limit is specified,
        # so need to check if the limit was specified
        if params is not None:
            if 'limit' in params:
                # limit=0 means grab everything, so ignore the limit parameter
                # in that case.
                if params['limit'] != 0 and params['limit'] < total:
                    logging.info("A limit was specified, so adjusting total " +
                                 "to " + str(params['limit']))
                    return params['limit']

        return total

    def query_collection(self, collection, params=None):
        url = self.base_url + "/" + collection
        return self.query(url, params)

    def query_long(self, url, params):
        
        if params is None:
            raise SCOTExtractorException("Called query_long without params " +
                "defined")
        if params is not None and 'id' not in params:
            raise SCOTExtractorException("Called query_long without id list " +
                "defined")

        results = []
        alert_ids = params['id']
        i = 0
        tmpList = []
        for alert_id in alert_ids:
            tmpList.append(int(alert_id))
            i = i + 1
            if i >= ID_LIST_LENGTH_LIMIT:
                smallParams = dict(params)
                smallParams['id'] = tmpList
                otherresults = self.query(url, smallParams)
                if otherresults is not None:
                    results = results + otherresults
                i = 0
                tmpList = []

        # Finish up the query
        if i < ID_LIST_LENGTH_LIMIT and i > 0:
            smallParams = dict(params)
            smallParams['id'] = tmpList
            otherresults = self.query(url, smallParams)
            if otherresults is not None:
                results = results + otherresults

        return results


    def modify_alert_field(self, alert_id, key, value):

        url = self.base_url + "/" + ALERT_COLLECTION  + "/" + str(alert_id)
        data = {key : value}
        self.session.put(url, verify=False, data=data)

    def query(self, url, params=None):
        """ Queries the restful interface at the specified url and parameters.
        """

        if params == None:
            params = {}


        if ('id' in params and isinstance(params['id'], list) and 
            len(params['id']) > ID_LIST_LENGTH_LIMIT ):

            logging.info("The list of ids is too long; spliting into multiple" +
                     " queries: " + str(len(params['id']))+ " vs " + 
                     str(ID_LIST_LENGTH_LIMIT))
            return self.query_long(url, params)

        # Getting the number of records to be retrieved. 
        totalCount = self.get_record_count(url, params)

        # If a limit was supplied in the parameters, we'll get that many
        # records instead of the total count for the query without the limit.
        # limit as a parameter is later overwritten because we perform the query
        # incrementally, so we need to capture the desired limit before we write
        # over it.
        increment = self.increment
        if 'limit' in params:
            if params['limit'] < increment:
                # A limit of 0 actually means grab everything
                if params['limit'] != 0:
                    increment = params['limit']

            if totalCount > params['limit']:
                # A limit of 0 actually means grab everything
                if params['limit'] != 0:
                    totalCount = params['limit']

        queryCount = 0
        offset = 0
        records = None

        while queryCount < totalCount:
            params['offset'] = offset
            params['limit'] = increment 
            response = None
            max_num_tries = 10
            worked = False
            try_num = 0
            while not worked and try_num < max_num_tries:
                worked = True
                try:
                    response = self.session.get(url, verify=False,
                                            headers=self.headers, params=params)
                except ConnectionError:
                    worked = False
                    logging.warn("Connection Exception in query.")
                    if response is not None:
                        logging.warn("Request url " + response.url)
                    else:
                        logging.warn("Request url " + url + " params " + 
                                     str(params))
 
                if response is not None and response.status_code == 200:
                    response_records = None
                    try:
                        response_records = response.json()['records']
                    except JSONDecodeError as e:
                        logging.exception(str(e))
                        logging.error("Error decoding json.")
                        worked = False 
                    if records == None:
                        records = response_records 
                    else:
                        if isinstance(records, dict):
                            records.update(response_records)
                        elif isinstance(records, list):
                            records = records + response_records
                        else:
                            logging.error("Records is an unsupported type")
                else:
                    logging.error("Error with query")
                    worked = False

                try_num = try_num + 1
                if not worked:
                    logging.warn("Trying " + str((max_num_tries - try_num -1)) +
                                 " more times")

            if try_num >= max_num_tries:
                raise SCOTExtractorException("Couldn't connect") 
           
            offset = offset + self.increment
            queryCount = queryCount + self.increment

        return records
        

    '''
        Performs queries of the form: collection1/id/collection2
        which returns items from collection2 related to item with id of 
        collection1
    '''
    def get_related_items(self, collection1, item_id, collection2, params={}):
        url = self.base_url + "/" + collection1 + "/" + str(item_id)
        url = url + "/" + collection2
        return self.query(url, params)


    '''
        Gets entities of a particular type from a dictionary 
        of entities.  Some queries return a list and others return a 
        a dictionary.  I'm not sure why.  This method expects a dictionary
        and only those entities of type t are returned.
        \param entities A dictionary containing entities.
        \param t The type of entity we are looking for.  
    '''
    def get_entity_type_from_dictionary(self, entities, t):
        matches = {}
        if entities != None:
            for entity in entities:
                if entities[entity]["type"] == t:
                    matches[entity] = entities[entity]

        return matches
