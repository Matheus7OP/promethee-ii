from configparser import ConfigParser

import numpy as np

from .dataSupplier import DataSupplier
from time import time


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
        print()
        inicio = time()

        self.load_data()
        self.normalize_criteria()
        self.get_pairwise_comparisons()
        self.get_preference_degrees()
        self.get_global_preferences()
        self.get_partial_outranking_flows()
        self.get_net_flow()        
        results = self.obtain_results()

        final = time()
        print("Tempo de execução: " + str( "%.f" % ((final - inicio) * 1000.0 )) + " ms" )
        print("\n")
        
        return results

    def load_data(self, sample_size=50, use_columns=None):
        use_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] # or use_columns
        ds = DataSupplier(self.config)
        #                                    get_alternatives(number=None, columns=None, sample_seed=1, rnd=True)
        self.alternatives, self.indices = ds.get_alternatives(self.sample_size, use_columns, self.seed, self.rnd)
        self.alternatives_number, self.criteria_number = self.alternatives.shape
        self.criteria_weights = np.ones(self.criteria_number) / self.criteria_number

        print("Quantidade de criterios: " + str(self.criteria_number))
        print ("Quantidade de alternativas: " + str( len(self.indices) ))
        #print ("Alternativas: " + str(self.alternatives))

    def normalize_criteria(self):
        self.normalized_alternatives = self.alternatives / self.alternatives.max(axis=0)

    def get_pairwise_comparisons(self):
        self.pairwise_comparisons = np.zeros((self.alternatives_number, self.alternatives_number, self.criteria_number),
                                             dtype=np.float)

        #print("Pairwise_comparisons antes de calcular: ")
        #print(self.pairwise_comparisons)

        for i in range(self.alternatives_number):
            for j in range(self.alternatives_number):
                self.pairwise_comparisons[i, j] = self.normalized_alternatives[i] - self.normalized_alternatives[j]

        #print("Pairwise_comparisons depois   de calcular: ")
        #print(self.pairwise_comparisons)

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
                #print(self.preference_degrees[i, j])
                #print(np.dot(self.preference_degrees[i, j], self.criteria_weights))
                self.global_preferences[i, j] = np.dot(self.preference_degrees[i, j], self.criteria_weights)

    def get_partial_outranking_flows(self):
        self.positive_outranking_flow = np.sum(self.global_preferences, axis=1) / (self.alternatives_number - 1)
        self.negative_outranking_flow = np.sum(self.global_preferences, axis=0) / (self.alternatives_number - 1)

    def get_net_flow(self):
        self.net_flow = self.positive_outranking_flow - self.negative_outranking_flow

    def obtain_results(self):
        results = list(zip(self.indices, self.net_flow))
        results = sorted(results, key=lambda result: result[1], reverse=True)
        
        return results

    def read_config(self, config_filename):
        config = ConfigParser()
        config.read(config_filename)
        return config
