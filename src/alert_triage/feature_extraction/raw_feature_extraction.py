"""Extract features from raw alert

This class extracts features from the body of an alert.

"""

import datetime
import re
import logging
import pickle
from alert_triage.database import database
from alert_triage.feature_extraction.abstract_feature_extraction import (
    AbstractFeatureExtraction)

_DEBUG = False
_email_search = re.compile('[\w\.-]+@[\w\.-]+').search


class RawFeatureExtractionException(Exception):

    """Exception type for the RawFeatureExtraction class."""

    pass


class RawFeatureExtraction(AbstractFeatureExtraction):

    """Extract features from raw alert"""

    def __str__(self):
        return "RawFeatureExtraction"

    def pickle(self, handle):
        """ Nothing to be done for the custom pickle function.  Nothing is
            written.
        """
        pass

    def unpickle(self, handle):
        """ Nothing to be done for the custom unpickle function.  Nothing is
            read.
        """
        pass

    
    def build_model(self):
        """ This class does not have an associated model, so a no-op function
        """
        pass

    def extract_features(self, alert):
        """Extract features"""
        logging.debug("getting features for alert " + str(alert.get("id")))

        alert_id = alert.get("id")
        if not alert_id:
            return

        data = alert.get("data")

        features = {}
        if not data:
            return features

        features.update(self._extract_blocked_features(data))
        features.update(self._extract_email_features(data))
        features.update(self._extract_file_features(data))
        features.update(self._extract_generic_features(alert, data))
        features.update(self._extract_http_features(data))
        features.update(self._extract_malware_features(data))
        features.update(self._extract_time_features(alert))
        return features

    def _extract_email_features(self, data):
        """Extract raw features related to email"""
        features = {}
        line_count = data.get("linecount")
        mail_from = data.get("MAIL_FROM")
        mime_type = data.get("mime_type")
        reached_exchange = data.get("REACHED_EXCHANGE")
        subject = data.get("MAILSUBJECT")

        if line_count:
            features["email_line_count"] = int(line_count)

        if mail_from:
            mail = False
            mail = _email_search(mail_from)
            if mail:
                features["mail_from"] = mail.group(0)

        if mime_type:
            if mime_type == '-':
                mime_type = "unknown"
            mime_type = re.sub("[ \-,\/\']", '_', mime_type).lower()
            features["email_mime_type"] = mime_type

        if reached_exchange:
            if reached_exchange == "TRUE":
                features["email_reached_exchange"] = 1
            elif reached_exchange == "FALSE":
                features["email_reached_exchange"] = 0

        if subject:
            fwd_cnt = subject.count("Fwd:")
            fwd_cnt += subject.count("FWD:")
            fwd_cnt += subject.count("FW:")
            re_cnt = subject.count("Re:")
            re_cnt += subject.count("RE:")
            features["email_fwd_count"] = fwd_cnt
            features["email_re_count"] = re_cnt

        return features

    def _extract_file_features(self, data):
        features = {}

        computer_name = data.get("COMPUTER_NAME")
        app_name = data.get("APP_Name")
        operation_sys = data.get("OPERATION_SYSTEM")

        def _file_feature(file_string, designator, features):
            designator = "file_" + designator
            if file_string:
                file_string = re.sub("[ \-,\/\']", '_', file_string).lower()
                features[designator] = file_string
            return features
        features = _file_feature(computer_name, "computer_name", features)
        features = _file_feature(app_name, "app_name", features)
        features = _file_feature(operation_sys, "operation_system", features)
        return features

    def _extract_blocked_features(self, data):
        logging.info("Entering _extract_blocked_features")
        features = {}

        source = data.get("badscriptsource")

        if source:
            source = re.sub("[ \-,\/\']", '_', source).lower()
            features["blocked_bad_script_source"] = source

        return features

    def _extract_generic_features(self, alert, data):
        """Extract generic raw features"""
        features = {}

        source = alert.get("source")
        index = data.get("index")

        if index:
            index = re.sub("[ \-,\/\']", '_', index).lower()
            features['index'] = index

        if source:
            # convert to ascii to make our lives easier
            source = source.encode("ascii", "ignore")
            # Do substring matches here.
            analyst_str = '[A-Za-z ]+[,][ A-Za-z]+'

            def _get_source(source_str, source_name, source):
                source_reg = re.compile(source_str).search
                if source_reg(source):
                    source = source_name
                return source

            if ',' in source:
                source = _get_source(analyst_str, 'analyst', source)
            elif 'ForefrontServerProtectionEXCH' in source:
                source = 'forefront'
            elif 'FireEye' in source:
                source = 'fireeye'
            elif 'MailGate' in source:
                source = 'mailgate'
            elif 'MDS' in source:
                source = 'mds'
            elif 'Symantec' in source:
                source = 'symantec'
            features['source'] = source

        return features

    def _extract_http_features(self, data):
        """Extract raw features related to HTTP addresses"""
        features = {}

        method = data.get("method")
        os = data.get("os")
        owner = data.get("owner")
        request_body_len = data.get("request_body_len")
        response_body_len = data.get("response_body_len")
        status_code = data.get("status_code")
        user_agent = data.get("user_agent")
        if user_agent is None:
            user_agent = data.get("short_usragt")
        methods = ['get', 'head', 'options', 'post', 'put', 'propfind',
                   'ntloaddriver', 'connect']
        oses = ['screenos', 'juniper', 'linux', 'macos', 'windows']

        def _check_for_unknown(known_ids, identifier, feature_string,
                               features):
            feature_identifier = False
            if identifier:
                identifier = identifier.lower()
                for item in known_ids:
                    if identifier.__contains__(item):
                        feature_identifier = feature_string + item
                        features[feature_identifier] = 1
                        return features
                feature_identifier = feature_string + 'unknown'
                features[feature_identifier] = 1
            return features

        features = _check_for_unknown(methods, method, 'http_method_',
                                      features)
        features = _check_for_unknown(oses, os, 'http_os_',
                                      features)

        if owner:
            if "VPN" in owner:
                features["http_owner_unknown"] = 1
            else:
                owner = re.sub("[\ \.\-\"\']", '_', owner).lower()
                features["http_owner"] = owner

        if request_body_len:
            request_body_len = int(request_body_len)
            if request_body_len == 0:
                features["http_request_body_length_zero"] = 1
            features["http_request_body_length"] = int(request_body_len)

        if response_body_len:
            response_body_len = int(response_body_len)
            if response_body_len == 0:
                features["http_response_body_length_zero"] = 1
            features["http_response_body_length"] = int(response_body_len)

        if status_code:
            if status_code.isdigit():
                features["http_status_code"] = int(status_code)
            else:
                features["http_status_code_unknown"] = 1
        agents = ['google', 'java', 'perl', 'php', 'python']
        browsers = ['chrome', 'firefox', 'opera', 'safari']
        ua_oses = ['linux', 'mac', 'msie', 'windows']

        def _agent_check(agent_list, user_agent, feature_string, features):
            if user_agent:
                for agent in agent_list:
                    if agent in user_agent.lower():
                        feature_identifier = feature_string + agent
                        features[feature_identifier] = 1
            return features
        features = _agent_check(agents, user_agent, 'http_user_agent_',
                                features)
        features = _agent_check(browsers, user_agent,
                                'http_user_agent_browser_', features)
        features = _agent_check(ua_oses, user_agent, 'http_user_agent_os_',
                                features)
        if user_agent:
            user_agent_lowercase = user_agent.lower()

            # mobile devices
            # - order matters: ipad user agents contain "iphone" in the
            # but iphone user agents don't contain "ipad" in the string
            if "ipad" in user_agent_lowercase:
                features["http_user_agent_mobile_ipad"] = 1
            elif "iphone" in user_agent_lowercase:
                features["http_user_agent_mobile_iphone"] = 1

            # TODO: user agent: extract browser versions, os versions

        return features

    def _extract_malware_features(self, data):
        """Extract raw features related to malware"""
        features = {}

        current_login_user = data.get("CURRENT_LOGIN_USER")
        malware_detected = data.get("malware-detected")
        service_pack = data.get("SERVICE_PACK")
        sid = data.get("sid")
        vlan = data.get("vlan")

        if current_login_user:
            current_login_user = re.sub("[ \.\-,\/\']", '_',
                                        current_login_user).lower()
            features["malware_current_login_user"] = current_login_user

        if malware_detected or malware_detected == "":
            features['malware_detected'] = 1

        # security identifier
        if sid:
            if sid.isdigit():
                features["malware_sid"] = sid
            else:
                features["malware_sid_unknown"] = 1

        if service_pack:
            service_pack = service_pack.strip()
            if service_pack:
                service_pack = service_pack.encode("ascii", "ignore")
                service_pack = re.sub("[ \.\-,\/\']", '_', service_pack).lower()
                features["malware_windows"] = service_pack

        if vlan:
            vlan = vlan.strip()
            features["malware_vlan"] = vlan

        return features

    def _extract_time_features(self, alert):
        """Extract raw features related to time"""
        features = {}

        time = alert.get("created")
        if time:
            # assume the first history array entry is the alert creation time
            #creation = history[0]['when']
            date_creation = datetime.datetime.utcfromtimestamp(time)
            dates = map(int, str(date_creation).split()[0].split('-'))
            # set the time features
            features['time_year'] = dates[0]
            features['time_month'] = dates[1]
            features['time_day'] = dates[2]
            features['time_weekday'] = date_creation.isoweekday()

        return features

"""
 This hasn't been refactored and probably doesn't work
"""
class SplunkSearchFeatureExtraction(object):
    def __init__(self, database_name):
        self._database_name = database_name
        self._splunk_search_map = {}

    def __str__(self):
        return "SplunkSearchFeatureExtraction"

    def build_model(self):
        connection = database.Database(database=self._database_name)
        self._splunk_search_map = {None : "splunk_search_unknown"}
        count = 0
        for rule_text in connection.db.alerts.distinct("data.search"):
            self._splunk_search_map[rule_text] = ("splunk_search_" +
                str(count))
            count += 1

    def extract_features(self, alert):
        features = {}
        rule_text = alert.get("data.search", None)
        key = self._splunk_search_map[rule_text]
        features[key] = 1
        return features
