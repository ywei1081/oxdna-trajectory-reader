from .oxdna_trajectory_reader import read_configurations, read_indicies
from .configuration import Configuration, ConfigurationSlice, Nucleotide, dumps_configurations
from .trajectory import TrajReader
from .topology import Topology

NM_PER_UNIT_LENGTH = 0.8518


__all__ = [
    'read_configurations',
    'read_indicies',
    'dumps_configurations',
    'Configuration',
    'ConfigurationSlice',
    'Nucleotide',
    'TrajReader',
    'Topology',
    'NM_PER_UNIT_LENGTH',
]
