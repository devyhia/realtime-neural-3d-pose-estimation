import numpy as np
from models.classifier import NearestNeighbour


def test_match():
    database = np.array([
        [2, 2],
        [1, 1],
        [0,0],
        [-2, -2]
    ])

    targets = [ 'P1', 'P2', 'P3', 'P4' ]

    clf = NearestNeighbour(database, targets)

    assert clf.match(np.array([.75, .75])) == 'P2'
    assert clf.match(np.array([1.75, 1.75])) == 'P1'
    assert clf.match(np.array([-1, -1])) == 'P3'
    assert clf.match(np.array([-1.01, -1.01])) == 'P4'