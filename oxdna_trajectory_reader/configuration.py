from __future__ import annotations
import typing

import numpy as np
import numpy.typing as npt

from .oxdna_trajectory_reader import dumps_configurations as _dumps_configurations


def dumps_configurations(configurations: list[Configuration]) -> list[str]:
    """
    Convert `Configuration`s back to strings using trajectory file format

    :param configurations: List of `Configuration`
    :return: List of string per configuration
    """
    if not all(isinstance(c, Configuration) for c in configurations):
        other_types = set(type(c).__name__ for c in configurations if not isinstance(c, Configuration))
        raise TypeError(f'All elements in configurations must be Configuration, not {", ".join(other_types)}')
    return _dumps_configurations([(c.time, c.box, c.energy, c._nucleotides) for c in configurations])


class ConfigurationBase:
    def __init__(self, time: int, box: npt.NDArray[np.float64],
                 energy: npt.NDArray[np.float64], nucleotides: npt.NDArray[np.float64]):
        self.time = time
        self.box = box
        self.energy = energy
        self._nucleotides = nucleotides

    @property
    def positions(self):
        return self._nucleotides[:, :3]

    @property
    def a1s(self):
        return self._nucleotides[:, 3:6]

    @property
    def a3s(self):
        return self._nucleotides[:, 6:9]

    @typing.overload
    def __getitem__(self, index: int) -> Nucleotide:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> ConfigurationSlice:
        ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return ConfigurationSlice(self.time, self.box, self.energy, self._nucleotides[index])
        elif isinstance(index, int):
            return Nucleotide(self.time, self.box, self.energy, self._nucleotides[index])
        else:
            raise ValueError(f"Invalid index type: {type(index)}")


class Configuration(ConfigurationBase):
    """
    A configuration frame of the system

    :param time: int
    :param box: 1x3 np.array
    :param energy: 1x3 np.array
    :param nucleotides: Nx15 np.array
    """
    def to_str(self) -> str:
        """Convert configuration to string using trajectory file format"""
        return _dumps_configurations([(self.time, self.box, self.energy, self._nucleotides)])[0]


class ConfigurationSlice(ConfigurationBase):
    pass


class Nucleotide:
    def __init__(self, time: int, box: npt.NDArray[np.float64],
                 energy: npt.NDArray[np.float64], nucleotide: npt.NDArray[np.float64]):
        self._time = time
        self._box = box
        self._energy = energy
        self._nucleotide = nucleotide

    @property
    def position(self):
        return self._nucleotide[:3]

    @property
    def a1(self):
        return self._nucleotide[3:6]

    @property
    def a3(self):
        return self._nucleotide[6:9]
