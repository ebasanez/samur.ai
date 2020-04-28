import time
import random

import utils

class TrafficManager():
    def __init__(self, update_period, districts):
        self.update_period = update_period
        self.last_update = time.time()
        self.districts = districts
        self.traffic = {district : 0 for district in districts.keys()}

    def _update_traffic(self):
        self.traffic = {district : random.uniform(0, 100) for district in self.traffic.keys()}

    def get_traffic_per_district(self):
        if time.time() - self.last_update > self.update_period:
            self._update_traffic()

        return self.traffic

    def get_displacement_time(self, distance_per_district):
        if time.time() - self.last_update > self.update_period:
            self._update_traffic()

        total_time = sum([distance / utils.get_speed(self.traffic[district]) for district, distance in distance_per_district.items()])

        return total_time