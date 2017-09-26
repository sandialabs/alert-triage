"""Database operations

This class provides an interface for working with a database backend.

"""


import time

import pymongo

DEBUG = True


def labeled_alert_query(alert_database, limit=0):
    # Find promoted alert ids 0 = all
    promoted_cursor = alert_database.find().limit(int(limit / 2))
    promoted_ids = [item["alert_id"] for item in promoted_cursor]
    closed_cursor = alert_database.find().limit(abs(limit - len(promoted_ids)))
    closed_ids = [item["alert_id"] for item in closed_cursor]
    all_ids = promoted_ids + closed_ids
    # Mongo query format -- { $in: [<value1>, <value2>, ... <valueN> ] }
    return {"alert_id": {"$in": all_ids}}


def unlabeled_alert_query(alert_database, limit=0, date=None):
    # { $or: [ { <expression1> }, { <expression2> }, ... , { <expressionN> } ]}
    if date is not None:
        unlabeled_cursor = alert_database.find().limit(limit)
        all_ids = [item["alert_id"] for item in unlabeled_cursor]
    return {"alert_id": {"$in": all_ids}}


class Database(object):
    def __init__(self, slave=False, database=False,
                 collection='alerts', port=27017):

        """Create an instance of the Database class

        arguments:
            database: name of the database to connect
            port: the database port
            slave: whether this database is a slave instance
            collection: the collection in the database to eventually query

        """
        self.database = database
        self.collection = collection

        if slave:
            readpref = pymongo.ReadPreference.SECONDARY
        else:
            readpref = pymongo.ReadPreference.PRIMARY

        self._db_connection = pymongo.MongoClient(port=port,
                                                  read_preference=readpref)
        self.db = self._db_connection[self.database]

    def find(self, *args, **kargs):
        """Query the database.

        Wrapper to pymongo.find. A limit of zero is equivalent to no limit.

        """
        collection = kargs.pop('collection', self.collection)
        if not collection:
            collection = self.collection
        return self.db[collection].find(*args, **kargs)

    def insert(self, d, **kargs):
        """ Wrapper to pymongo insert. """
        collection = kargs.pop('collection', self.collection)
        if not collection:
            collection = self.collection
        return self.db[collection].insert(d, **kargs)

    def sort(self, *args, **kargs):
        """ Wrapper to pymongo sort. """
        collection = kargs.pop('collection', self.collection)
        if not collection:
            collection = self.collection
        return self.db[collection].sort(*args, **kargs)

    def update(self, *args, **kargs):
        """ Wrapper for pymongo update. """
        collection = kargs.pop('collection', self.collection)
        if not collection:
            collection = self.collection
        return self.db[collection].update(*args, **kargs)

    def write(self, data, collection=None):
        """Take an JSON alert data source and write it to the database.

        For use with the json_update code

        """
        if not collection:
            collection = self.collection
        self.db[collection].insert(data)

    def write_labels(self, labels, collection=None):
        """Writes all alerts and labels to the database"""
        if not collection:
            collection = self.collection
        for alert, value in labels.items():
            self.db[collection].insert({str(int(alert)): value})

    def read_labels(self, collection=None):
        """Read labels from a collection"""
        if not collection:
            collection = self.collection
        labels = {}
        all_labels = self.db[collection].find()
        for item in all_labels:
            del item['_id']
            labels.update(item)
        return dict(zip(map(int, labels.keys()), labels.values()))

    def delete_collection(self, collection=None):
        """Delete a collection from the database"""
        if not collection:
            collection = self.collection
        self.db[collection].drop()

    def close(self):
        """ Close the connection to the database. """
        self._db_connection.close()

    def add_ranking(self, alert_id, ranking):
        """Add rankings to each alert_id"""
        t = time.time()
        self.update(
            {
                "id": alert_id
            },
            {
                "$push": {
                    "rankings": {"date": t, "ranking": ranking}
                }
            },
            True
        )

    def get_last_sent_ranking(self, alert_id):
        """Determine the last ranking received"""
        rankings = self.find({"alert_id": alert_id})
        try:
            return rankings[0]["rankings"][-1]["ranking"]
        except IndexError:
            return None
