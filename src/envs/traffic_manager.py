
from datetime import datetime, timedelta
import random

class TrafficManager():

    def __init__(self, start_time, districts, 
        update_period: int = 9000,
        max_avg_speed: float = 60.0,
        max_load: float = 100.0,
        ):

        self.update_period = timedelta(seconds=update_period)
        self.max_avg_speed = max_avg_speed
        self.max_load = max_load

        self.last_update = start_time
        self.districts = districts
        self.traffic = {district : 0 for district in districts.keys()}

    def update_traffic(self, time):
        """Provisionally, traffic at each district and update period is sampled from a random
        distribution between 0 and 100 traffic load.
        """
        if (time - self.last_update) > self.update_period: 
            self.traffic = {district : random.triangular(5, self.max_load -5, 30) 
                            for district in self.traffic.keys()}
            self.last_update = time

    def _get_speed(self, traffic_load):
        return self.max_avg_speed * (1 - traffic_load / self.max_load)

    def displacement_time(self, distance_per_district, time):
        self.update_traffic(time)

        # If something is outside the limits, it gets assigned average traffic of present districts
        other_districts_traffic = [self.traffic[district] 
                                   for district in distance_per_district.keys() 
                                   if district != 'Missing']
        self.traffic['Missing'] = sum(other_districts_traffic) / len(other_districts_traffic)

        total_time = sum([distance / self._get_speed(self.traffic[district]) 
                          for district, distance in distance_per_district.items()]) * 3600

        return total_time

    def set_time(self, time):
        self.last_update = time

