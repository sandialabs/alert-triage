"""Run the near realtime pipeline.

This class waits for alerts to stream into SCOT
then classifies and ranks the alert using the 
0model built by the batch pipeline.

NearRealtimePipelineException: this is the exception
time that should be raised when exceptions or errors
occur.

NearRealtimePipeline: this is the only class defined
by this module. Its run_pipeline handles the continuous
operation of the pipeline.

"""

import json
import pickle
import numpy
import time

import alert_triage
from alert_triage.util import scot_helper
from alert_triage.feature_extraction.scot_extractor import SCOTExtractor
from stomp import *

_DEBUG = True
_BASE_URL = scot_helper.BASE_URL

class NearRealtimePipelineException(Exception):

    """Exception type for the Near-Realtime Pipeline Class"""

    pass


class NearRealtimePipeline(object):

    """Run the near-realtime pipeline.

    This class runs in the background.

    run_pipeline(): this method runs the pipeline.

    """

    def __init__(self, testing=False):
        """Initialize the NearRealtime class.

        Arguments:
            testing:
                A boolean that if true indicates
                pipeline in test mode

        """
        self._testing = testing
	"""Load AMQ parameters"""
	self._message_queue_ip = scot_helper.AMQ
	self._message_queue_port = scot_helper.AMQ_PORT
	self._message_queue_destination = scot_helper.AMQ_DEST
	
    def run_pipeline(self, stop_time=600.):
	if self._testing == True:
	    stop_time = 2400.
	self._listen_in_background(stop_time)

    def stop_pipeline(self):
	pass

    def _listen_in_background(self, stop_time):
	"""Listen to AMQ until stop_time"""
	class Listener(object):
	    """Listener class for handling queue"""
	    msg = ""
	    def __init__(self):
                self.msg = ""
	    def on_message(self, headers, message):
		self.msg = message
	"""Initialize listener and subscribe to AMQ"""
	conn = Connection([(self._message_queue_ip, int(self._message_queue_port))])
	listener = Listener()
	conn.set_listener('', listener)
	conn.start()
	conn.connect(wait=True)
	conn.subscribe(destination='/topic/scot', id=1000, ack='auto')
	msg = ""
	"""Initialize the batch mode models"""
	#self._load_models()
	start_time = time.time()
	end_time = start_time + stop_time
	self._scot = SCOTExtractor(_BASE_URL) 
	while time.time() < end_time:
	    """Until stop time, run listen to AMQ in background"""
	    if msg != listener.msg:
		"""When a new message arrive, trigger"""
		msg = listener.msg
		"""Turn messages into alerts"""
		alerts = self._process_message(msg)
		print alerts
		#if len(alerts) > 0:
		#    for alert in alerts:
		#	"""Turn alerts in rankings"""
		#	self._rank_alerts(alert)
	conn.disconnect()

    def _process_message(self, msg):
	data = json.loads(msg)
	alerts = []
	if data["action"] == "created":
	    print data
	    if data["data"]["type"] == "alert":
	        alert_id = data["data"]["id"]
	        alerts.append(alert_id)
	return alerts

    def _pull_alert(self, alert_id):
	pass

    def _load_models(self):
	'''Waiting on Eric to handle these'''
	pass

    def _send_ranking(self, alert_data):
	pass

    def _rank_alert(self, alert):
	pass
