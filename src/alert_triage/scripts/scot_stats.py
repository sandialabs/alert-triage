
import ConfigParser
import argparse
from alert_triage.feature_extraction import scot_extractor
from alert_triage.feature_extraction.scot_extractor import SCOTExtractor



def main():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--limit',
                         type=int,
                         default=100)
    parser.add_argument('--config',
                         type=str,
                         default="config.ini")
                         

    args = parser.parse_args()
    limit = args.limit

    scot = SCOTExtractor(baseUrl)

    params = {}
    if limit != 0:
        params['limit'] = limit

    alerts = scot.query_collection(scot_extractor.ALERT_COLLECTION,
                                   params)


    counts = {}
    for alert in alerts:
        key = alert['status']
        current = counts.get(key, 0)
        counts[key] = current + 1


    print "Total number of alerts", len(alerts)
    print counts

main()



