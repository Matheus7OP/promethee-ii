import random

import numpy as np
import pandas as pd


class DataSupplier(object):
    def __init__(self, config):
        self.config = config
        self.alternatives_df = None

    def load_alternatives(self, number=None, columns=None, sample_seed=123, rnd=True):
        alternatives_filename = 'insurance_data/ticdata2000.txt'
        self.alternatives_df = pd.read_table(alternatives_filename, header=None)
        if columns:
            self.alternatives_df = pd.DataFrame(self.alternatives_df, columns=columns)
        if rnd and number:
            self.alternatives_df = self.alternatives_df.head(number)
            return self.alternatives_df, np.arange(0, number)
        if not rnd and number:
            random.seed(sample_seed)
            rindex = np.array(random.sample(range(len(self.alternatives_df)), number))
            self.alternatives_df = self.alternatives_df.ix[rindex]
            return self.alternatives_df.values, rindex
        return self.alternatives_df.values

    def objectify(self, alts):
        alternatives = []
        for alt in alts:
            alternatives.append(Alternative(alt))
        return self.alternatives_df.values

    def get_alternatives(self, number=None, columns=None, sample_seed=1, rnd=True):
        alts = self.load_alternatives(number, columns, sample_seed, rnd)
        return self.objectify(alts), alts[1]


class Alternative(object):
    def __init__(self, alt):
        self.criteria = []
        self.add_criteria(alt)

    def add_criteria(self, values):
        for value in values:
            self.criteria.append(Criterion(value))

    def get_criteria(self):
        return np.array([criterion.value for criterion in self.criteria])


class Criterion(object):
    def __init__(self, value):
        self.value = value
