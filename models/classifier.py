import numpy as np

class NearestNeighbour(object):
    def __init__(self, database, targets):

        assert database.shape[0] == len(targets), \
            """
            Targets should map to database.
            Number of database objects (= {}) mismatch the number of targets (= {})""".format(
                database.shape[0], len(targets)
            )

        self.database = database
        self.targets = targets
    
    def match(self, item):
        assert item.shape[0] == self.database.shape[1], \
            "Item shape (= {}) mismatches the database items (= {})".format(
                item.shape[0], self.database.shape[1]
            )
        
        distances = (item - self.database)**2
        distances = distances.sum(axis=1)
        distances = np.sqrt(distances)

        nearest_neighbour = np.argmin(distances)

        return self.targets[nearest_neighbour]