import unittest
import logging
import numpy as np
from numpy.testing import assert_array_equal as assert_equal
from numpy.testing import assert_array_almost_equal as assert_almost_equal
from rescompy import plotter


__author__ = ['Daniel Canaday', 'Dayal Kalra', 'Alexander Wikner',
              'Declan Norton', 'Brian Hunt', 'Andrew Pomerance']
__version__ = '1.0.0'


logger = logging.getLogger()
logger.setLevel(logging.ERROR)