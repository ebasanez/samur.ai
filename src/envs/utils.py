import math

MAX_LOAD = 100 # load ranges from 0 to 100
MAX_AVG_SPEED = 50 # km/h
HOUR_IN_SECS = 3600 # seconds

# Calculates cartesian product of point(tuples) pairs
def cartesian(*kwargs):
    if len(kwargs) % 2 != 0:return 0
    result = 0
    for i in range(0, len(kwargs), 2):
        p1 = kwargs[i]
        p2 = kwargs[i + 1]
        result += math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    return result

def get_speed(traffic_load):
    return - HOUR_IN_SECS * traffic_load * MAX_AVG_SPEED / MAX_LOAD + MAX_AVG_SPEED # km/s