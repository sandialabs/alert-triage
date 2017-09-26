"""Perform STOMP and JSON related tasks

This class interacts with a STOMP and JSON server to send and receive SCOT
data.  The data is written to database when received.  Also has a listing
function that triggers a callback when the STOMP server notifies that the
alert has been changed.

JSONupdate: This has the methods that interact with the JSON and STOMP
servers.

"""

import base64
import json
import urllib2
import ast
import time
from stompy.simple import Client

from alert_triage.database import database

_DEBUG = True

class JSONupdate(object):

    """Interact with JSON and STOMP servers

    This class provides functions that interact with the JSON and STOMP
    servers.

    clone_db(): This method clones multiple collections within a database.
    The set of collections is passed as a dictionary with keys being the
    collection name, and values being the max id within that collection name.
    For example, {'alerts': 40} will clone the alerts database up to alert_id
    40.  The rate limit paramater gives a time in seconds between calls to the
    json api.

    clone_collection(): This method clones a single collection within a
    database.  It uses the same paramaters as clone_db, except the input is
    not a dictionary.

    poll(): This function will put the execution in a blocking state until a
    new message is passed via the stomp server.  When the new message is
    received, the callback function is executed.

    json_request(): This function will make a json request to the main
    database server.  The data of this json request is returned as a dict.

    callback(): This function is executed after an incoming message from
    the stomp server is received.  If the message is not of type 'view', then
    it will perform a json_request for the updated data.

    """

    def __init__(self, scot_uri, scot_database='scot3'):
        """Create an instance of the JSONupdate class

        arguments:
            stomp_uri: the base uri to the scot stomp server.
            stomp_port: the port used for communicating with the scot stomp
                server.  Defaults to 61613
            scot_database: the scot database to write new alert data to.
                Defaults to scot_v3

        """
        self._scot_uri = scot_uri
        self._stomp_uri = stomp_uri
        self._stomp_port = stomp_port
        self._db = database.Database(database=scot_database,
                                     collection='alerts')

        self._stomp_client = Client(host=self._stomp_uri, port=self._stomp_port)
        self._stomp_client.connect()
        self._stomp_client.subscribe('/topic/CHAT.DEMO',
                                     conf={'message':'chat', 'type':'listen'})

    def clone_db(self, collection_list, rate_limit=3):
        """clone multiple collections from the main scot server via JSON

        arguments:
            collection_list: A dictionary of each collection to clone.  The
                keys are collection names, and the keys are the max id within
                that collection.
            rate_limit: Time in seconds to wait between each connection
                attempt to the JSON server.  Defaults to 3.

        """
        for collection in collection_list:
            self.clone_collection(collection, collection_list[collection],
                                  rate_limit)

    def cdb(self, entity_type, s1, s2, rate_limit=3):
        """clone multiple collections from the main scot server via JSON

        arguments:
            collection_list: A dictionary of each collection to clone.  The
                keys are collection names, and the keys are the max id within
                that collection.
            rate_limit: Time in seconds to wait between each connection
                attempt to the JSON server.  Defaults to 3.

        """
        for id in xrange(s1, s2):
            data = self.json_request(entity_type, id)
            self._db.write(data, collection=entity_type)

            if _DEBUG:
                print "added id:", str(id), "entity_type:", entity_type,
                time.sleep(rate_limit)


    def clone_collection(self, entity_type, max_id, rate_limit=3):
        """clone a single collection from the main scot server via JSON

        arguments:
            entity_type: the name of the collection to clone
            max_id: The value of the last id to clone
            rate_limit: Time in seconds to wait between each connection
                attempt to the JSON server.  Defaults to 3.

        """
        for id in range(1, max_id):
            data = self.json_request(entity_type, id)
            self._db.write(data, collection=entity_type)
            if _DEBUG:
                print "id:", str(id), "entity_type:", entity_type
            time.sleep(rate_limit)


    def cc(self, entity_type, max_id, rate_limit=3):
        """clone a single collection from the main scot server via JSON

        arguments:
            entity_type: the name of the collection to clone
            max_id: The value of the last id to clone
            rate_limit: Time in seconds to wait between each connection
                attempt to the JSON server.  Defaults to 3.

        """
        for id in range(1, max_id):
            data = self.json_request(entity_type, id)
            print data
            if _DEBUG:
                print "entity_type:", entity_type
                print "id:", str(id)
            time.sleep(rate_limit)

    def poll(self):
        """Block and wait until a message is recieved from the STOMP server"""
        self._stomp_client.get(block=True, callback=self.callback)

    def json_request(self, entity_type, id):
        """Retrieve data from the JSON server

        arguments:
            entity_type: The name of the collection to get data from
            id: The id value of the data to get

        """
        print "json_request"
        url = self._scot_uri + str(entity_type) + '/' + str(id)

    def callback(self, frame):
        """Request JSON data after a STOMP message has been received

        arguments:
            frame: A STOMPY frame that contains thet data sent over the
                STOMP channel

        """
        print "callback"
        body = frame.body

        print frame
        #print type(body)
        print "----"


        #Conversion into python data structure
        body = body.replace('null', 'None')
        #change from str to dict
        body = ast.literal_eval(body)


        print "action:", body['action']
        print "type:", body['type']
        if body['action'] != 'view': #ignore views as nothing has changed.
            type = body['type']
            id = body['id']
            if _DEBUG:
                print 'STOMP Body {1}'.format(body)
                print '{1} number {2}'.format(type, id)
            self._db.write(self.json_request(type, id))
