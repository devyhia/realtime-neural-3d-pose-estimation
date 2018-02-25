import numpy as np

class Quaternion(object):
    """
    Quaternions represent a rotation matrix in pose estimation applications
    Source: https://de.mathworks.com/help/aerotbx/ug/quatmultiply.html
    """

    def __init__(self, a,b,c,d):
        """A quaternion is described as follows:
            Quaternion = a + b (i) + c (j) + d (k)
            where (i), (j) and (k) are direction vectors.
        
        Arguments:
            a {float} -- constant
            b {float} -- value in the direction (i)
            c {float} -- value in the direction (j)
            d {float} -- value in the direction (k)
        """

        self.q = (a,b,c,d)
    
    def dot(self, other):
        assert type(other) is Quaternion, \
            "Quanternions are only multiplied by other quanternions!"
        
        n0 = other.q[0] * self.q[0] - other.q[1] * self.q[1] - other.q[2] * self.q[2] - other.q[3] * self.q[3]
        n1 = other.q[0] * self.q[1] + other.q[1] * self.q[0] - other.q[2] * self.q[3] + other.q[3] * self.q[2]
        n2 = other.q[0] * self.q[2] + other.q[1] * self.q[3] + other.q[2] * self.q[0] - other.q[3] * self.q[1]
        n3 = other.q[0] * self.q[3] - other.q[1] * self.q[2] + other.q[2] * self.q[1] + other.q[3] * self.q[0]

        return Quaternion(n0, n1, n2, n3)
    
    def distance(self, other):
        assert type(other) is Quaternion, \
            "Quanternions are only multiplied by other quanternions!"
        
        q1 = np.array(self.q)
        q2 = np.array(other.q)
        
        return 2 * np.arccos(min(1, np.abs(q1.dot(q2.T))))

    def __mul__(self, other):
        return self.dot(other)
    
    def __str__(self):
        return "Quaternion= ({}, {}, {}, {})".format(*self.q)
    
    def __repr__(self):
        return "Quaternion({}, {}, {}, {})".format(*self.q)

    def __eq__(self, other):
        return self.q == other.q