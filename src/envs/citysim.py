#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=unbalanced-tuple-unpacking
"""
Gym environment for simulating the development of emergencies requiring ambulance in a city.

This environment will be used in order to train a resurce allocation agent. The agent receives
information about emergencies and their severity. The agent can send an ambulance to attend the
emergency from one of the different hospitals in the city, and make the ambulance go back to the
original hospital or to one of the other hospitals in the city.

If not responded to in time, the emergencies can result in failure situations, sampled from a
probability distribution according to their severity.

Emergencies are gerated from representative probability distributions.

Created by Enrique Basañez, Miguel Blanco, Alfonso Lagares, Borja Menéndez and Francisco Rueda.
"""

import calendar
import os
from collections import defaultdict, deque, namedtuple
from datetime import datetime, timedelta
from pathlib import Path
import geopandas as gpd
from shapely.geometry import (
    Polygon,
    Point,
    MultiPolygon,
    LinearRing,
    MultiLineString,
    MultiPoint,
    LineString,
)

import gym
import numpy as np
import yaml
from gym import spaces
from gym.utils import seeding
from recordclass import recordclass

from .traffic_manager import TrafficManager


class CitySim(gym.Env):
    """Gym environment for simulating ambulance emergencies in a city.

    Attributes:
        city_config: str or Path, YAML file with parameters describing the city to simulate. If no 
            file is provided, default values will be used.
        timestep: float, time advanced at eachc step should be high to avoid sparse actions but low
            to enable accuracy. Compromise. One minute by default.
        stress: float, multiplier for the emergency generator, in order to artificially increase or
            decrease the amount of emergencies and modify the stress to the system.
    """

    metadata = {
        "render.modes": ["rgb_array", "console"],
        "video.frames_per_second": 30,
    }

    def __init__(
        self,
        city_config="data/city_defaults.yaml",  # YAML file w/ city and generator data
        city_geometry="data/madrid_districs_processed/madrid_districs_processed.shp",
        time_step: int = 60,
        stress: float = 1.0,
    ):
        """Initialize the CitySim environment."""
        assert os.path.isfile(city_config), "Invalid path for city configuration file"
        assert os.path.isfile(city_geometry), "Invalid path for city geometry file"

        self.time_step_seconds = time_step
        self.time_step = timedelta(seconds=self.time_step_seconds)
        self.stress = stress

        # Named lists for status keeping
        self.hospital = recordclass("Hospital", ["name", "loc", "available_amb"])
        self.emergency = recordclass("Emergency", ["loc", "severity", "tappearance"])
        self.moving_amb = recordclass(
            "MovingAmbulance", ["tobjective", "thospital", "destination", "reward"]
        )

        # Read configuration file for setting up the city
        if type(city_config) is dict:
            config = city_config
        else:
            with open(city_config) as config_file:
                config = yaml.safe_load(config_file)
        geometry = gpd.read_file(city_geometry)
        self._configure(config, geometry)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(
        self,
        time_start: datetime = datetime.fromisoformat("2020-01-01T00:00:00"),
        time_end: datetime = datetime.fromisoformat("2024-12-31T23:59:59"),
    ):
        """Return the environment to the start of a new scenario, with no active emergencies. 

        Same city, but start and end times can be different. This is a necessary step at the start.
        """
        self.time_start = time_start
        self.time_end = time_end

        self.time = self.time_start
        self.active_emergencies = ["dummy"] + [deque() for i in range(self.severity_levels)]
        self.outgoing_ambulances = []
        self.incoming_ambulances = []

        self.traffic_manager = TrafficManager(self.time, self.districts)

        for i in self.hospitals.keys():
            self.hospitals[i]["available_amb"] = self.config["hospitals"][i]["available_amb"]

        return self._get_obs()

    def step(self, action):

        # Advance time
        self.time += self.time_step
        self.traffic_manager.update_traffic(self.time)

        # Generate new emergencies. Emergencies are a series of FIFO lists, one per severity
        self._generate_emergencies()

        # Check for objectives in outgoing ambulances and apply failures to reward
        new_outgoing = []
        reward = 0
        for ambulance in self.outgoing_ambulances:
            if self.time >= ambulance["tobjective"]:
                reward += ambulance["reward"]
                self.incoming_ambulances.append(ambulance)
            else:
                new_outgoing.append(ambulance)
        self.outgoing_ambulances = new_outgoing

        # Check for final destinations in incoming ambulances and add them to the roster
        new_incoming = []
        for ambulance in self.incoming_ambulances:
            if self.time >= ambulance["thospital"]:
                self.hospitals[ambulance["destination"]]["available_amb"] += 1
            else:
                new_outgoing.append(ambulance)
        self.incoming_ambulances = new_incoming

        # Take actions. As many actions as (hospitals + 1) X severity categories X hospitals
        start_hospitals, end_hospitals = action
        for severity, queue in enumerate(self.active_emergencies):
            start_hospital_id, end_hospital_id = start_hospitals[severity], end_hospitals[severity]
            start_hospital = self.hospitals[start_hospital_id]
            end_hospital = self.hospitals[end_hospital_id]
            if severity == 0:  # Dummy severity to move ambulances between hospitals
                if (end_hospital_id == 0) or (start_hospital_id == 0):
                    continue  # Null hospitals would not make sense here
                self.hospitals[start_hospital_id]["available_amb"] -= 1
                tthospital = self._displacement_time(start_hospital["loc"], end_hospital["loc"])
                ambulance = self.moving_amb(self.time, self.time + tthospital, end_hospital_id, 0)
                self.incoming_ambulances.append(ambulance)
                continue

            if start_hospital_id == 0:  # Starting hospital #0 simbolizes null action
                continue
            if len(queue) == 0:  # If the queue for this severity level is empty, no action
                continue
            if start_hospital["available_amb"] == 0:  # No ambulances, no action
                continue

            if end_hospital_id == 0:  # Null end hospital to return to start hospital
                end_hospital_id = start_hospital_id
                end_hospital = start_hospital

            # Launch an ambulance from start hospital towards emergency
            self.hospitals[start_hospital_id]["available_amb"] -= 1
            emergency = self.active_emergencies[severity].popleft()
            ttobj = self._displacement_time(start_hospital["loc"], emergency["loc"])
            tthospital = self._displacement_time(emergency["loc"], end_hospital["loc"]) + ttobj
            time_diff = -ttobj
            ambulance = self.moving_amb(
                self.time + ttobj,
                self.time + tthospital,
                end_hospital_id,
                self._reward_f(time_diff, severity),
            )

            self.outgoing_ambulances.append(ambulance)

        # Return state, reward, and whether the end time has been reached
        return self._get_obs(), reward, self.time >= self.time_end, {}

    def render(self, mode="console"):
        print(self._get_obs())

    def close(self):
        pass

    def set_stress(self, stress):
        """Modify the stress factor at any moment in the execution."""
        self.stress = stress

    def _configure(self, config, geometry):
        """Set the city information variables to the configuration."""

        self.config = config

        self.hospitals = config["hospitals"]
        self.districts = config["districts"]
        self.severity_levels = config["severity_levels"]
        self.severity_dists = config["severity_dists"]
        self.shown_emergencies_per_severity = config["shown_emergencies_per_severity"]

        self.geo_df = geometry.rename(columns={"district_c": "district_code"}).set_index(
            "district_code"
        )

        # Correct possible discrepancies in hospital district data and geometry data
        for hospital_id, hospital in self.hospitals.items():
            point = Point(hospital["loc"]["x"], hospital["loc"]["y"])
            hospital_district_code = 0
            for district_code, row in self.geo_df.iterrows():
                polygon = row["geometry"]
                if polygon.contains(point):
                    hospital_district_code = district_code
            self.hospitals[hospital_id]["loc"]["district_code"] = hospital_district_code

    def _get_obs(self):
        """Build the part of the state that the agent can know about.

        This includes hospital locations, ambulance locations, incoming emergencies.
        """

        observation = []

        # Hospitals table
        # id x y available_amb incoming_amb ttamb
        hospitals_table = []
        for id, hospital in self.hospitals.items():
            x, y, district = hospital["loc"]
            incoming = 0
            for ambulance in self.outgoing_ambulances + self.incoming_ambulances:
                if ambulance["destination"] == id:
                    incoming += 1
            hospital_data = [id, x, y, district, hospital["available_amb"], incoming]
            hospitals_table.append(hospital_data)
        observation.append(np.array(hospitals_table))

        # Unattended emergencies, with locations and severity. 3D table in severity/order/data
        # Data for each emergency is severity order time_active x y
        emergencies_table = []
        for severity, queue in enumerate(self.active_emergencies):
            severity_table = []
            if severity == 0:
                continue
            for order in range(self.shown_emergencies_per_severity):
                if order < len(queue):
                    emergency = queue[order]
                    x, y, district = emergency["loc"]
                    tactive = int((self.time - emergency["tappearance"]) / self.time_step)
                    emergency_data = [severity, order, tactive, x, y, district]
                else:
                    emergency_data = [0, order, 0, 0, 0, 0]
                severity_table.append(emergency_data)
            emergencies_table.append(severity_table)
        observation.append(np.array(emergencies_table))

        # Districts data?

        # Time data
        time_data = np.array(
            [
                self.time_step_seconds,  # Information about potential reaction time
                self.time.month,
                self.time.day,
                self.time.weekday() + 1,
                self.time.hour,
                self.time.minute,
            ]
        )
        observation.append(time_data)

        return observation

    def _generate_emergencies(self):
        """For given city parameters and time, generate appropriate emergencies for a timestep.

        Emergencies come predefined with the time to failure, which is softly correlated to severity.

        The agent only knows about the location, severity and the time since it was generated.
        """

        hour = self.time.hour
        weekday = self.time.weekday() + 1
        month = self.time.month

        for severity in range(1, self.severity_levels + 1):
            base_frequency = self.severity_dists[severity]["frequency"]
            current_frequency = (
                base_frequency
                * self.severity_dists[severity]["hourly_dist"][hour]
                * self.severity_dists[severity]["daily_dist"][weekday]
                * self.severity_dists[severity]["monthly_dist"][month]
                * self.stress
            )

            # Assuming independent distributions per hour, weekday and month
            period_frequency = current_frequency * self.time_step_seconds  # Avg events per step

            # Poisson distribution of avg # of emergencies in period will give number of new ones
            num_new_emergencies = int(np.random.poisson(period_frequency, 1))

            if num_new_emergencies == 0:
                continue

            # Get the district weights for the current severity
            probs_dict = self.severity_dists[severity]["district_prob"]
            district_weights = np.array([w for district, w in sorted(probs_dict.items())])
            district_weights = district_weights / district_weights.sum()

            for _ in range(num_new_emergencies):  # Skipped if 0 new emergencies
                district = np.random.choice(  # District where emergency will be located
                    np.arange(len(district_weights)) + 1, p=district_weights
                )
                loc = self._random_loc_in_distric(district)
                tappearance = self.time
                emergency = self.emergency(loc, severity, tappearance)
                self.active_emergencies[severity].append(emergency)  # Add to queue

    def _displacement_time(self, start, end):
        """Given start and end points, returns a displacement time between both locations for an 
        ambulance, based on the current traffic, metheorology, and randomness.

        (x1, y1, district1) (x2, y2, district2)  [km], centro P. del Sol, x -> Este, y -> Norte
        """

        distance_per_district = self._get_segments_per_district(
            start["district_code"],
            (start["x"], start["y"]),
            end["district_code"],
            (end["x"], end["y"]),
        )
        total_time = self.traffic_manager.displacement_time(distance_per_district, self.time)

        return timedelta(seconds=total_time)

    def _get_random_point_in_polygon(self, polygon):
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
            if polygon.contains(point):
                return point

    def _random_loc_in_distric(self, district_code):
        polygon = self.geo_df.loc[district_code]["geometry"]
        point = self._get_random_point_in_polygon(polygon)
        x, y = np.array(point.coords).flatten().tolist()
        return {"x": x, "y": y, "district_code": district_code}

    def _obtain_route_cuts(self, origin, destination):
        route = LineString([Point(origin[0], origin[1]), Point(destination[0], destination[1])])

        cuts = {}  # Dict with {district_code: list of (x, y) tuples},
        for district_code, district_row in self.geo_df.iterrows():
            distric_geo = district_row["geometry"]
            # To allow handling more than 2 intersection points between route line and polygon
            distric_lr = LinearRing(list(distric_geo.exterior.coords))
            intersection = distric_lr.intersection(route)
            # Intersects district in ONE point
            if type(intersection) == Point:
                cuts[district_code] = [(intersection.x, intersection.y)]
            # Intersects district in TWO OR MORE points
            if type(intersection) == MultiPoint:
                cuts[district_code] = [(point.x, point.y) for point in intersection]
        return cuts

    # Calculate distances traversed across districts
    def _get_segments_per_district(
        self, district_origin, origin, district_destination, destination
    ):
        cuts = self._obtain_route_cuts(origin, destination)
        # If may return empty for same origin and destination district, but it will need an entry
        if len(cuts) == 0:
            cuts[district_origin] = []

        # Add points at start and end corresponding to the origin and destination
        cuts[district_origin].insert(0, origin)
        cuts[district_destination].append(destination)

        distances = {k: self._cartesian(*v) for (k, v) in cuts.items()}
        distances["Missing"] = self._cartesian(origin, destination) - sum(distances.values())

        return distances

    def _cartesian(self, *kwargs):
        """Given a series of points in th Cartesian plane, returns the sums of the distances 
        between consecutive pairs of such points. There must be an even number of points.
        """
        if len(kwargs) % 2 != 0:
            return 0
        result = 0
        for i in range(0, len(kwargs), 2):
            p1 = kwargs[i]
            p2 = kwargs[i + 1]
            result += np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        return result

    def _reward_f(self, time_diff, severity):
        """Possible non-linear fuction to apply to the time difference between an ambulance arrival
        and the time reference of the emergency in order to calculate a reward for the agent.
        """
        return time_diff.seconds * severity  # Right now linear with time to emergency and severity
