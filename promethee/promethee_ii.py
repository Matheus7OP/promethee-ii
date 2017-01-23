from configparser import ConfigParser

import numpy as np

from promethee.dataSupplier import DataSupplier


class PrometheeII(object):
    def __init__(self, config_filename='default.conf', sample_size=50, seed=123, weights=None, rnd=True):
        self.config = self.read_config(config_filename)
        self.alternatives = None
        self.alternatives_number = 0
        self.criteria_weights = weights
        self.criteria_number = 0
        self.normalized_alternatives = None
        self.pairwise_comparisons = None
        self.preference_degrees = None
        self.global_preferences = None
        self.positive_outranking_flow = None
        self.negative_outranking_flow = None
        self.net_flow = None
        self.results = None
        self.sample_size = sample_size
        self.seed = seed
        self.indices = None
        self.rnd = rnd

    def run(self):
        self.load_data()
        self.normalize_criteria()
        self.get_pairwise_comparisons()
        self.get_preference_degrees()
        self.get_global_preferences()
        self.get_partial_outranking_flows()
        self.get_net_flow()
        return self.obtain_results()

    def load_data(self, sample_size=50, use_columns=None):
        # use_columns = use_columns or [1, 18, 31, 32, 38, 39, 40, 42, 67]
        ds = DataSupplier(self.config)
        self.alternatives, self.indices = ds.get_alternatives(self.sample_size, use_columns, self.seed, self.rnd)
        self.alternatives_number, self.criteria_number = self.alternatives.shape
        # self.criteria_weights = np.ones(self.criteria_number) / self.criteria_number

    def normalize_criteria(self):
        self.normalized_alternatives = self.alternatives / self.alternatives.max(axis=0)

    def get_pairwise_comparisons(self):
        self.pairwise_comparisons = np.zeros((self.alternatives_number, self.alternatives_number, self.criteria_number),
                                             dtype=np.float)
        for i in range(self.alternatives_number):
            for j in range(self.alternatives_number):
                self.pairwise_comparisons[i, j] = self.normalized_alternatives[i] - self.normalized_alternatives[j]

    def get_preference_degrees(self):
        self.preference_degrees = np.zeros_like(self.pairwise_comparisons, dtype=np.uint8)
        for i in range(self.alternatives_number):
            for j in range(self.alternatives_number):
                self.preference_degrees[i, j][self.pairwise_comparisons[i, j] <= 0] = 0
                self.preference_degrees[i, j][self.pairwise_comparisons[i, j] > 0] = 1

    def get_global_preferences(self):
        self.global_preferences = np.zeros((self.alternatives_number, self.alternatives_number), dtype=np.float)
        for i in range(self.alternatives_number):
            for j in range(self.alternatives_number):
                self.global_preferences[i, j] = np.dot(self.preference_degrees[i, j], self.criteria_weights)

    def get_partial_outranking_flows(self):
        self.positive_outranking_flow = np.sum(self.global_preferences, axis=1) / (self.alternatives_number - 1)
        self.negative_outranking_flow = np.sum(self.global_preferences, axis=0) / (self.alternatives_number - 1)

    def get_net_flow(self):
        self.net_flow = self.positive_outranking_flow - self.negative_outranking_flow

    def obtain_results(self):
        results = list(zip(self.indices, self.net_flow))
        results = sorted(results, key=lambda result: result[1], reverse=True)
        print(results)
        return results

    def read_config(self, config_filename):
        config = ConfigParser()
        config.read(config_filename)
        return config
