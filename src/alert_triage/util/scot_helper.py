"""
"""

import base64
import json
import urllib2
from datetime import date
from dateutil.relativedelta import relativedelta


#_read_file = raw_input("Enter parameter filename:")
_read_file = 'config.ini'
_PARAMETERS = {}
with open(_read_file) as fp:
    for line in fp:
        (param, value) = line.strip().split('\t', 1)
        _PARAMETERS[param] = value

_DEBUG = True
PROXY_URL = _PARAMETERS['PROXY_URL']
NUM_ALERTS_TESTING = 10000
ALL_ALERTS = 0
SCOT_DEV_SYNC_URL = ""
SCOT_QUAL_SYNC_URL = ""
SCOT_QUAL = ""
SCOT_QUAL_ALERT_UPDATE_URL = ""
SCOT_SYNC_URL = ""
SCOT_ALERT_UPDATE_URL = ""
AMQ = ""
AMQ_PORT = ""
AMQ_DEST = ""
ALERTGROUP_URL = ""
ENTITY_URL = ""
BASE_URL = ""
LABEL_WEIGHTS = {"false_positive": 0,
                 "open_not_viewed": None,
                 "open_viewed": None,
                 "revisit": None,
                 "promoted_false_positive": 1,
                 "promoted": 1,
                 "incident": 1
                 }
IPV4 = _PARAMETERS['IPV4']
IPV4_PRIORITY = _PARAMETERS['IPV4_PRIORITY']
IPV4_THREAT = _PARAMETERS['IPV4_THREAT']
IPV4_2 = _PARAMETERS['IPV4_2']
IPV4_2_THREAT = _PARAMETERS['IPV4_2_THREAT']
DOMAIN = _PARAMETERS['DOMAIN']
DOMAIN_THREAT = _PARAMETERS['DOMAIN_THREAT']
DOMAIN_2 = _PARAMETERS['DOMAIN_2']
DOMAIN_2_THREAT = _PARAMETERS['DOMAIN_2_THREAT']
DOMAIN_PRIORITY = _PARAMETERS['DOMAIN_PRIORITY']
AMQ = _PARAMETERS['AMQ']
AMQ_PORT = _PARAMETERS['AMQ_PORT']
AMQ_DEST = _PARAMETERS['AMQ_DEST']
ALERT_GROUP_URL = _PARAMETERS['alertgroup_url']
ENTITY_URL = _PARAMETERS['entity_url']
BASE_URL = _PARAMETERS['base_url']

def seconds_epoch(original_date = date.today(), num_months = 3):
    if original_date != date.today():
        original_date = date.fromtimestamp(original_date)
    num_months_ago = original_date - relativedelta(months=num_months)
    return (num_months_ago - date(1970, 1, 1)).total_seconds()

def scot_put(alert_id, json_data):
    proxy_handler = urllib2.ProxyHandler({"https": PROXY_URL})
    https_handler = urllib2.HTTPSHandler()
    opener = urllib2.build_opener(proxy_handler, https_handler)

    url = SCOT_QUAL_ALERT_UPDATE_URL + str(alert_id)

    request = urllib2.Request(url, data=json_data)
    request.get_method = lambda: "PUT"
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", str(len(json_data)))

    encoded_pw = base64.encodestring().replace('\n', '')
    request.add_header("Authorization", "Basic %s" % encoded_pw)

    return urllib2.urlopen(request)

def scot_get_modified_since(last_modified_time):
    urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler({"https":PROXY_URL})))

    url = SCOT_QUAL_SYNC_URL + str(last_modified_time)

    request = urllib2.Request(url)

    encoded_pw = base64.encodestring().replace('\n', '')
    request.add_header("Authorization", "Basic %s" % encoded_pw)

    return json.load(urllib2.urlopen(request))
