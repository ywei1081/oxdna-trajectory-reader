from __future__ import annotations
import typing

import numpy as np

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
    def __init__(self, time: int, box: npt.NDArray[np.float64], energy: npt.NDArray[np.float64],
                 nucleotides: npt.NDArray[np.float64], backbone='oxDNA1'):
        self.time = time
        self.box = box
        self.energy = energy
        self._nucleotides = nucleotides
        self._backbone = backbone
        self._a2s = None
        self._base_end = None
        self._base_center = None
        self._backbone_center = None

    def copy(self):
        return self.__class__(self.time, self.box.copy(), self.energy.copy(), self._nucleotides.copy(), self._backbone)

    def rotate(self, rotation_matrix: npt.NDArray[np.float64], rotation_center: npt.NDArray[np.float64] | None = None):
        """rotate and modify configuration using rotation matrix"""
        if not rotation_matrix.shape == (3, 3):
            raise ValueError(f"rotation_matrix shape must be (3, 3), not {rotation_matrix.shape}")
        if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=1e-4):
            raise ValueError("rotation_matrix should have determinant=1.0")
        if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-4):
            raise ValueError("rotation_matrix should be orthogonal, ie R @ R.T = I")

        rotation_center = rotation_center if rotation_center is not None else np.zeros(3)
        if not rotation_center.shape == (3,):
            raise ValueError(f"rotation_center shape must be (3,), not {rotation_center.shape}")

        self.positions = (rotation_matrix @ (self.positions - rotation_center).T).T + rotation_center
        self.a1s = (rotation_matrix @ self.a1s.T).T
        self.a3s = (rotation_matrix @ self.a3s.T).T
        self._nucleotides[:, 9:12] = (rotation_matrix @ self._nucleotides[:, 9:12].T).T
        self._nucleotides[:, 12:15] = (rotation_matrix @ self._nucleotides[:, 12:15].T).T

    @property
    def positions(self):
        """nucleotide mass center"""
        return self._nucleotides[:, :3]

    @positions.setter
    def positions(self, value):
        self._nucleotides[:, :3] = value
        self._base_end = self._base_center = self._backbone_center = None

    @property
    def a1s(self):
        """base vectors a1"""
        return self._nucleotides[:, 3:6]

    @a1s.setter
    def a1s(self, value):
        self._nucleotides[:, 3:6] = value
        self._a2s = self._base_end = self._base_center = self._backbone_center = None

    @property
    def a3s(self):
        """base normal vectors a3"""
        return self._nucleotides[:, 6:9]

    @a3s.setter
    def a3s(self, value):
        self._nucleotides[:, 6:9] = value
        self._a2s = self._backbone_center = None

    @property
    def backbone(self):
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        self._backbone = value
        self._backbone_center = None

    @property
    def a2s(self):
        if self._a2s is None:
            self._a2s = np.cross(self.a3s, self.a1s)
        return self._a2s

    @property
    def base_end_positions(self):
        """base end / nucleotide end at base direction"""
        if self._base_end is None:
            self._base_end = self.positions + self.a1s * 0.6
        return self._base_end

    @property
    def base_center_positions(self):
        """base centroid / hydrogen-bonding/repulsion site"""
        if self._base_center is None:
            self._base_center = self.positions + self.a1s * 0.4
        return self._base_center

    @property
    def backbone_center_positions(self):
        """backbone centroid / backbone repulsion site"""
        if self._backbone_center is None:
            if self._backbone == 'oxDNA2':
                self._backbone_center = self.positions - 0.34 * self.a1s + 0.3408 * self.a2s
            elif self._backbone == 'RNA':
                self._backbone_center = self.positions + self.a1s * -0.4 + self.a3s * 0.2
            else:
                self._backbone_center = self.positions + self.a1s * -0.4
        return self._backbone_center

    @typing.overload
    def __getitem__(self, index: int) -> Nucleotide:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> ConfigurationSlice:
        ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return ConfigurationSlice(self.time, self.box, self.energy, self._nucleotides[index], self._backbone)
        elif isinstance(index, int):
            return Nucleotide(self.time, self.box, self.energy, self._nucleotides[index], self._backbone)
        else:
            raise ValueError(f"Invalid index type: {type(index)}")


class Configuration(ConfigurationBase):
    """
    A configuration frame of the system

    :param time: int
    :param box: 1x3 np.array
    :param energy: 1x3 np.array
    :param nucleotides: Nx15 np.array
    :param backbone: model geometry to use to calculate backbone center, 'oxDNA1', 'oxDNA2', or 'RNA'
    """
    def to_str(self) -> str:
        """Convert configuration to string using trajectory file format"""
        return _dumps_configurations([(self.time, self.box, self.energy, self._nucleotides)])[0]


class ConfigurationSlice(ConfigurationBase):
    pass


class Nucleotide:
    def __init__(self, time: int, box: npt.NDArray[np.float64],
                 energy: npt.NDArray[np.float64], nucleotide: npt.NDArray[np.float64],
                 backbone='oxDNA1'):
        self._time = time
        self._box = box
        self._energy = energy
        self._nucleotide = nucleotide
        self._backbone = backbone

    def copy(self):
        return Nucleotide(self._time, self._box.copy(), self._energy.copy(), self._nucleotide.copy(), self._backbone)

    @property
    def position(self):
        return self._nucleotide[:3]

    @position.setter
    def position(self, value):
        self._nucleotide[:3] = value

    @property
    def a1(self):
        return self._nucleotide[3:6]

    @a1.setter
    def a1(self, value):
        self._nucleotide[3:6] = value

    @property
    def a3(self):
        return self._nucleotide[6:9]

    @a3.setter
    def a3(self, value):
        self._nucleotide[6:9] = value

    @property
    def a2(self):
        return np.cross(self.a3, self.a1)

    @property
    def base_end(self):
        return self.position + self.a1 * 0.6

    @property
    def base_center(self):
        return self.position + self.a1 * 0.4

    @property
    def backbone_center(self):
        if self._backbone == 'oxDNA2':
            return self.position - 0.34 * self.a1 + 0.3408 * self.a2
        elif self._backbone == 'RNA':
            return self.position + self.a1 * -0.4 + self.a3 * 0.2
        else:
            return self.position + self.a1 * -0.4


if typing.TYPE_CHECKING:
    import numpy.typing as npt
