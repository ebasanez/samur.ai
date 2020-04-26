#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gym environment for simulating the development of emergencies requiring ambulance in a city.

This environment will be used in order to train a resurce allocation agent. The agent receives
information about emergencies and their gravity. The agent can send an ambulance to attend the
emergency from one of the different hospitals in the city, and make the ambulance go back to the
original hospital or to one of the other hospitals in the city.

If not responded to in time, the emergencies can result in failure situations, sampled from a
probability distribution according to their gravity.

Emergencies are gerated from representative probability distributions.

Created by Enrique Basañez, Miguel Blanco, Alfonso Lagares, Borja Menéndez and Francisco Rueda.
"""

import calendar
from collections import defaultdict, deque, namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import gym
import numpy as np
import yaml
from gym import spaces
from gym.utils import seeding
from recordclass import recordclass


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
    defaults = {
        "hospitals": {
            0: {"name": "AAA", "x": 0.0, "y": 0.0},
            1: {"name": "Alcalá de Henares (Ppe. de Asturias)", "x": 29.814, "y": 10.59},
            2: {"name": "Central de la Defensa", "x": -3.849, "y": -2.765},
            3: {"name": "Clínico San Carlos", "x": -1.593, "y": 2.937},
            4: {"name": "Concepción (Fund. J. Díaz)", "x": -1.596, "y": 2.718},
            5: {"name": "Doce de Octubre", "x": 0.147, "y": -4.271},
            6: {"name": "Doctor Rodriguez Lafora", "x": 0.786, "y": 13.054},
            7: {"name": "Getafe", "x": -3.506, "y": -11.325},
            8: {"name": "Gregorio Marañón", "x": 2.59, "y": 0.523},
            9: {"name": "Infanta Leonor", "x": 7.038, "y": -3.105},
            10: {"name": "La Paz", "x": 1.13, "y": 7.416},
            11: {"name": "La Princesa", "x": 2.181, "y": 2.189},
            12: {"name": "Moncloa (ASISA)", "x": -2.914, "y": 2.06},
            13: {"name": "Niño Jesús (Infantil)", "x": 2.123, "y": 0.007},
            14: {"name": "Puerta de Hierro", "x": -14.652, "y": 3.992},
            15: {"name": "Ramón y Cajal", "x": 0.5, "y": 8.142},
            16: {"name": "Santa Cristina", "x": 2.553, "y": 0.821},
            17: {"name": "Virgen de la torre", "x": 6.956, "y": -3.685},
        },
        "districts": {
            1: {"name": "CENTRO", "surface": 5.21, "density": 25340.69},
            2: {"name": "ARGANZUELA", "surface": 6.52, "density": 23306.44},
            3: {"name": "RETIRO", "surface": 5.42, "density": 21867.53},
            4: {"name": "SALAMANCA", "surface": 5.36, "density": 26830.78},
            5: {"name": "CHAMARTIN", "surface": 9.12, "density": 15723.25},
            6: {"name": "TETUAN", "surface": 5.37, "density": 28664.25},
            7: {"name": "CHAMBERI", "surface": 4.73, "density": 29049.26},
            8: {"name": "FUENCARRAL", "surface": 238.0, "density": 1003},
            9: {"name": "MONCLOA", "surface": 46.47, "density": 2515.26},
            10: {"name": "LATINA", "surface": 25.47, "density": 9183.75},
            11: {"name": "CARABANCHEL", "surface": 14.1, "density": 17316.88},
            12: {"name": "USERA", "surface": 7.7, "density": 17535.32},
            13: {"name": "VALLECAS PTE.", "surface": 14.84, "density": 15345.01},
            14: {"name": "MORATALAZ", "surface": 6.08, "density": 15493.59},
            15: {"name": "CIUDAD LINEAL", "surface": 11.52, "density": 18455.56},
            16: {"name": "HORTALEZA", "surface": 25.87, "density": 6973.33},
            17: {"name": "VILLAVERDE", "surface": 20.21, "density": 7059.13},
            18: {"name": "VILLA DE VALLECAS", "surface": 51.49, "density": 2026.82},
            19: {"name": "VICALVARO", "surface": 35.36, "density": 1981.11},
            20: {"name": "SAN BLAS", "surface": 22.26, "density": 6934.37},
            21: {"name": "BARAJAS", "surface": 43.56, "density": 1076.06},
        },
        "emergency_levels": 5,
        "shown_emergencies_per_gravity": 20,
    }

    def __init__(
        self,
        city_config=None,  # YAML file w/ city and generator data
        time_step: timedelta = timedelta(seconds=60),
        stress: float = 1.0,
    ):
        """Initialize the CitySim environment."""

        self.time_step = time_step
        self.stress = stress

        # Named lists for status keeping
        self.hospital = recordclass("Hospital", ["name", "loc", "avail_amb"])
        self.emergency = recordclass("Emergency", ["loc", "gravity", "tappearance", "tfailure"])
        self.moving_amb = recordclass(
            "MovingAmbulance", ["tobjective", "thospital", "destination", "reward"]
        )

        # Read configuration file for setting up the city
        if city_config is None:
            self._configure(self.defaults)
        elif type(city_config) is dict:
            self._configure(city_config)
        else:
            with open(city_config) as config_file:
                config = yaml.safe_load(config_file)
                self._configure(config)

    def seed(self):
        pass

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
        self.active_emergencies = ["dummy"] + [deque() for i in range(self.emergency_levels)]
        self.outgoing_ambulances = []
        self.incoming_ambulances = []

        # TODO: Reset hospital available ambulances

    def step(self, action):

        # Advance time
        self.time += self.time_step

        # Generate new emergencies. Emergencies are a series of FIFO lists, one per gravity
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
                self.hospitals[ambulance["destination"]]["avail_amb"] += 1
            else:
                new_outgoing.append(ambulance)
        self.incoming_ambulances = new_incoming

        # Take actions. As many actions as (hospitals + 1) X gravity categories X hospitals
        start_hospitals, end_hospitals = action
        for gravity, queue in enumerate(self.active_emergencies):
            start_hospital = self.hospitals[start_hospitals[gravity]]
            end_hospital = self.hospitals[end_hospitals[gravity]]
            if gravity == 0:  # Dummy gravity to move ambulances between hospitals
                self.hospitals[start_hospital]["available_amb"] -= 1
                tthospital = self._displacement_time(start_hospital["loc"], end_hospital["loc"])
                ambulance = self.moving_amb(self.time, self.time + tthospital, end_hospital, 0)
                self.incoming_ambulances.append(ambulance)
                continue
            if len(queue) == 0:  # If the queue for this gravity level is empty, no action
                continue
            if start_hospital["name"] == "null":  # Starting hospital #0 simbolizes null action
                continue
            if start_hospital["avail_amb"] == 0:  # No ambulances, no action
                continue

            if end_hospital["name"] == "null":  # Null end hospital to return to start hospital
                end_hospital = start_hospital

            # Launch an ambulance from start hospital towards emergency
            self.hospitals[start_hospital]["avail_amb"] -= 1
            emergency = self.active_emergencies[gravity].popleft()
            ttobj = self._displacement_time(start_hospital["loc"], emergency["loc"])
            tthospital = self._displacement_time(emergency["loc"], end_hospital["loc"]) + ttobj
            time_diff = emergency["tfailure"] - (self.time + ttobj)
            ambulance = self.moving_amb(
                self.time + ttobj, self.time + tthospital, end_hospital, self._reward_f(time_diff)
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

    def _configure(self, config):
        """Set the city information variables to the configuration."""

        self.hospitals = config["hospitals"]
        self.districts = config["districts"]
        self.emergency_levels = config["emergency_levels"]
        self.shown_emergencies_per_gravity = config["shown_emergencies_per_gravity"]

    def _get_obs(self):
        """Build the part of the state that the agent can know about.

        This includes hospital locations, ambulance locations, incoming emergencies.
        """

        observation = []

        # Hospitals table
        # id x y avail_amb incoming_amb ttamb
        hospitals_table = []
        for id in self.hospitals:
            x, y = self.hospitals[id]["loc"]
            incoming = 0
            for ambulance in self.outgoing_ambulances + self.incoming_ambulances:
                if ambulance["destination"] == id:
                    incoming += 1
            hospital_data = [id, x, y, self.hospitals[id]["avail_amb"], incoming]
            hospitals_table.append(hospital_data)
        observation.append(np.array(hospitals_table))

        # Unattended emergencies, with locations and gravity. 3D table in gravity/order/data
        # Data for each emergency is gravity order time_active x y
        emergencies_table = []
        for gravity, queue in enumerate(self.active_emergencies):
            gravity_table = []
            if gravity == 0:
                continue
            for order in range(self.shown_emergencies_per_gravity):
                if order < len(queue):
                    emergency = queue[order]
                    x, y = emergency["loc"]
                    tactive = int((self.time - emergency["tappearance"]) / self.time_step)
                    emergency_data = [gravity, order, tactive, x, y]
                else:
                    emergency_data = [0, order, 0, 0, 0]
                gravity_table.append(emergency_data)
            emergencies_table.append(gravity_table)
        observation.append(np.array(emergencies_table))

        # Districts data?

        return observation

    def _generate_emergencies(self):
        """For given city parameters and time, generate appropriate emergencies for a timestep.

        Emergencies come predefined with the time to failure, which is softly correlated to gravity.

        The agent only knows about the location, gravity and the time since it was generated.
        """

        # por nivel de gravedad:
        #   distribución por hora (resolución del orden de hora)
        #   distribución por día de la semana
        #   distribución por semana del año
        #   distribución por distrito

        # DUMMY GENERATOR; TO BE COMPLETED
        for gravity in range(1, self.emergency_levels + 1):
            if np.random.rand() > 0.1 * self.stress:
                location = (1, 1)
                tapearance = self.time
                tfailure = self.time + timedelta(minutes=30)

                emergency = self.emergency(location, gravity, tapearance, tfailure)
                self.active_emergencies[gravity].append(emergency)
        pass

    def _displacement_time(
        self, start, end
    ):  # (x1, y1) (x2, y2)  [km], centro Puerta del Sol, x -> Este, y -> Norte
        """Given start and end points, returns a displacement time between both locations for an 
        ambulance, based on the current traffic, metheorology, and randomness.
        """

        # DUMMY GENERATOR; TO BE COMPLETED
        carthesian_distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
        speed = np.random.normal(1, 0.3)

        return carthesian_distance / speed  # Tiempo de desplazamiento [sec]

    def _reward_f(self, time_diff):
        """Possible non-linear fuction to apply to the time difference between an ambulance arrival
        and the time reference of the emergency in order to calculate a reward for the agent."""
        return time_diff  # Right now we will keep it linear with reaction time to emergency


city = CitySim()
f = open("city_defaults.yaml", "w+")
yaml.dump(city.defaults, f, allow_unicode=True)
