import sys
import os
sys.path.append(os.path.abspath('..'))
from alert_triage.database import database
from alert_triage.feature_extraction import label_extraction

_WEIGHTS = {"false_positive" : 0,
            "open_not_viewed" : None,
            "open_viewed" : None,
            "revisit" : None,
            "promoted_false_positive" : 0,
            "promoted" : 1,
            "incident" : 1
            }

def main():
    mongo = database.Database()
    labels = label_extraction.LabelExtraction(
        database=mongo, weights=_WEIGHTS).extract_labels()
    for rule_text in mongo.db.alerts.distinct("data.search"):
        tps = 0.0
        fps = 0.0
        count = 0.0
        for alert in mongo.db.alerts.find({"data.search": rule_text}):
            alert_id = alert["alert_id"]
            label = labels[alert_id]
            if label == 1:
                tps += 1.0
            elif label == 0:
                fps += 1.0
            count += 1.0
        print rule_text, count, tps/count, fps/count


if __name__ == "__main__":
    main()
