import numpy as np
from helpers.quaternion import Quaternion

q = Quaternion(*[1, 0, 1, 0])
q_same = Quaternion(*[1, 0, 1, 0])
r = Quaternion(*[1, 0.5, 0.5, 0.75])
n = Quaternion(0.5000, 1.2500, 1.5000, 0.2500)
q_times_q = Quaternion(*[0, 0, 2, 0])

def test_equality():
    assert q == q, "The object should equate itself"
    assert r == r, "The object should equate itself"
    assert n == n, "The object should equate itself"

    assert q == q_same, "The object should equate other object with same params"
    
    assert q != r, "The object should not equate others with different params"
    assert q != n, "The object should not equate others with different params"

def test_multiplication():
    assert q * r == n
    assert q * q == q_times_q

def test_theta():
    # Two far away objects
    q1 = (
        -0.28184579021235323,
        -0.6032481990846498,
        0.6534595646367771,
        -0.3600627142949052
    )
    r1 = (
        -0.09833017644372441,
        -0.5813174531644217,
        0.7703273692597304,
        -0.2428928554246895
    )
    q2 = (
        -0.2975296329306237,
        -0.5838329797881483,
        0.6765174897507843,
        -0.33606436184331906
    )

    prod = q1[0] * r1[0] + q1[1] * r1[1] + q1[2] * r1[2] + q1[3] * r1[3]
    print(prod)
    result = 2 * np.arccos(np.abs(prod))
    print(result)
    
    assert result == Quaternion(*q1).distance(Quaternion(*r1))
    assert result > 0.25 # asserts that angle between the two is greater than 10 degrees!

    # asserts that small differences in quaternions correspond to small angles/distances!
    assert Quaternion(*q1).distance(Quaternion(*q2)) < 0.25