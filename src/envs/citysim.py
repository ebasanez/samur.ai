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

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import namedtuple, defaultdict
from recordclass import recordclass
from datetime import datetime, timedelta
import calendar


class CitySim(gym.Env):
    """Gym environment for simulating ambulance emergencies in a city.
        
    Attributes:
        city_config: json file with parameters describing the city to simulate. If no file is 
            provided, default values will be used.
    """
    
    metadata = {
        'render.modes' : ['rgb_array', 'console'],
        'video.frames_per_second' : 30
    }
    
    defaults = {
        'hospitals' : {
            
        }
    }

    def __init__(self, 
                 city_config = None, 
                 time_step: timedelta = timedelta(seconds=60),  # Must be increased to avoid sparse actions but decreased to enable accuracy. Compromise 
                 ):
        """Initialize the CitySim environment."""
        
        self.time_step = time_step
        
        self.hospital = recordclass('Hospital', ['loc', 'avail_amb'])
        self.emergency = recordclass('Emergency', ['loc', 'gravity', 'tsappearance', 'ttfailure'])
        self.moving_amb = recordclass('Moving_Ambulance', ['ttobj', 'tthospital'])
        
        if city_config is not None:
            self._configure(city_config)
        else:
            self._configure(defaults)
    
    def seed(self):
        pass
    
    def reset(self,
              time_start: datetime = datetime.fromisoformat('2020-01-01T00:00:00'),
              time_end: datetime   = datetime.fromisoformat('2024-12-31T23:59:59'),
              ):
        """Return the environment to the start of a new scenario. 
        
        Same city, but start and end times can be different.
        """
        self.time_start = time_start
        self.time_end = time_end
        
        self.time = self.time_start
        
    
    def step(self, action):
        
        # Advance time
        self.time += self.time_step
        
        # Generate new emergencies. Emergencies are a series of FIFO lists, one per gravity (and district?)
        self._generate_emergencies()
        
        # Take actions. As many actions as hospitals X gravity categories (and districts?). There is a null action action
        
        # Advance counter for displacements
        
        # Check failure of emergencies
        
        # End the process if the end date was reached
        if self.time >= self.time_end:
            return self._get_obs(), reward, True, {}
        
        pass
    
    def render(self, mode='console'):
        pass
    
    def close(self):
        pass
    
    def _configure(self, config):
        """Set the city information variables to the configuration."""
        pass
    
    def _get_obs(self):
        """Build the part of the state that the agent can know about.
        
        This includes hospital locations, ambulance locations, incoming emergencies.
        """
        
        # Available ambulances per hospital
        
        # Hospital locations
        
        # Unattended emergencies, with locations and and gravity
        
        pass
    
    def _generate_emergencies():
        """For given city parameters and time, generate appropriate emergencies for a timestep.
        
        Emergencies come predefined with the time to failure, which is softly correlated to gravity.
        
        The agent only knows about the location, gravity and the time since it was generated.
        """
    