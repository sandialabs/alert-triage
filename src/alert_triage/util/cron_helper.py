"""
"""

import os

from alert_triage.util import filelock

MODIFIED_ALERTS_FILE = "/tmp/alert_triage_modified_alerts"

def read_modified_alert_ids():
    """ Read modified alert IDs from file, then remove them from the file."""
    # Return an empty list if the file doesn't exist.
    if not os.path.exists(MODIFIED_ALERTS_FILE):
        return []
    # Get a lock on the file
    lock = filelock.FileLock(MODIFIED_ALERTS_FILE, 5)
    lock.acquire()
    # Open the file and read in the data.
    fp = open(MODIFIED_ALERTS_FILE, "r+")
    ids = fp.read().split("\n")
    # remove zero length strings
    ids = filter(len, ids)
    # convert IDs to int
    ids = list(map(int, ids))
    # remove duplicates
    ids = list(set(ids))
    # close and remove the file
    fp.close()
    #TODO: uncomment when live
    #os.unlink(MODIFIED_ALERTS_FILE)
    # Release the lock.
    lock.release()
    return ids

def write_modified_alert_ids(ids):
    # Get a lock on the file
    lock = filelock.FileLock(MODIFIED_ALERTS_FILE, 5)
    lock.acquire()
    # Open the file and write the alert IDs.
    fp = open(MODIFIED_ALERTS_FILE, "a")
    for alert_id in ids:
        fp.write(str(alert_id) + "\n")
    fp.close()
    # Release the lock.
    lock.release()
