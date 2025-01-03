from __future__ import annotations
import functools
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


def readonly(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        array = func(*args, **kwargs)
        array.setflags(write=False)
        return array
    return wrapper


class Configuration:
    """
    A configuration frame of the system

    :param time: int
    :param box: 1x3 np.array
    :param energy: 1x3 np.array
    :param nucleotides: Nx15 np.array
    :param backbone_type: model geometry to use to calculate backbone center, 'oxDNA1', 'oxDNA2', or 'RNA'
    """

    def __init__(self, time: int, box: npt.NDArray[np.float64], energy: npt.NDArray[np.float64],
                 nucleotides: npt.NDArray[np.float64], backbone_type='oxDNA2'):
        self.time = time
        self.box = box
        self.energy = energy
        self._nucleotides = nucleotides
        self.backbone_type = backbone_type

    def to_str(self) -> str:
        """Convert configuration to string using trajectory file format"""
        return _dumps_configurations([(self.time, self.box, self.energy, self._nucleotides)])[0]

    def copy(self):
        return self.__class__(self.time, self.box.copy(), self.energy.copy(), self._nucleotides.copy(), self.backbone_type)

    @staticmethod
    def _check_valid_rotation_matrix(rotation_matrix: npt.NDArray[np.float64], atol=1e-4):
        if not rotation_matrix.shape == (3, 3):
            raise ValueError(f"rotation_matrix shape must be (3, 3), not {rotation_matrix.shape}")
        if not np.isclose(np.linalg.det(rotation_matrix), 1.0, atol=atol):
            raise ValueError("rotation_matrix should have determinant=1.0")
        if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=atol):
            raise ValueError("rotation_matrix should be orthogonal, ie R @ R.T = I")

    def rotate(self, rotation_matrix: npt.NDArray[np.float64], rotation_center: npt.NDArray[np.float64] | None = None):
        """rotate and modify configuration using rotation matrix"""
        self._check_valid_rotation_matrix(rotation_matrix)

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

    @property
    def a1s(self):
        """base vectors a1"""
        return self._nucleotides[:, 3:6]

    @a1s.setter
    def a1s(self, value):
        self._nucleotides[:, 3:6] = value

    @property
    def a3s(self):
        """base normal vectors a3"""
        return self._nucleotides[:, 6:9]

    @a3s.setter
    def a3s(self, value):
        self._nucleotides[:, 6:9] = value

    @property
    @readonly
    def a2s(self):
        return np.cross(self.a3s, self.a1s)

    @property
    @readonly
    def base_end_positions(self):
        """base end / nucleotide end at base direction"""
        return self.positions + self.a1s * 0.6

    @property
    @readonly
    def base_center_positions(self):
        """base centroid / hydrogen-bonding/repulsion site"""
        return self.positions + self.a1s * 0.4

    @property
    @readonly
    def backbone_center_positions(self):
        """backbone centroid / backbone repulsion site"""
        if self.backbone_type == 'oxDNA2':
            return self.positions - 0.34 * self.a1s + 0.3408 * self.a2s
        elif self.backbone_type == 'RNA':
            return self.positions + self.a1s * -0.4 + self.a3s * 0.2
        else:
            return self.positions + self.a1s * -0.4

    @typing.overload
    def __getitem__(self, index: int) -> Nucleotide:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> ConfigurationSlice:
        ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return ConfigurationSlice(self, *index.indices(len(self._nucleotides)))
        elif isinstance(index, int):
            if index >= len(self._nucleotides) or index < -len(self._nucleotides):
                raise IndexError(f"Index {index} out of bounds for configuration of length {len(self._nucleotides)}")
            if index < 0:
                index += len(self._nucleotides)
            return Nucleotide(self, index)
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

    def __iter__(self):
        for index in range(len(self._nucleotides)):
            yield Nucleotide(self, index)

    def __len__(self):
        return len(self._nucleotides)

    def __repr__(self):
        return f"<Configuration time={self.time} len={len(self)}>"


class ConfigurationSlice:
    def __init__(self, conf: Configuration, start: int, stop: int, step: int):
        self._conf = conf
        self._index = slice(start, stop, step)

    def rotate(self, rotation_matrix: npt.NDArray[np.float64], rotation_center: npt.NDArray[np.float64] | None = None):
        """rotate and modify configuration using rotation matrix"""
        Configuration._check_valid_rotation_matrix(rotation_matrix)

        rotation_center = rotation_center if rotation_center is not None else np.zeros(3)
        if not rotation_center.shape == (3,):
            raise ValueError(f"rotation_center shape must be (3,), not {rotation_center.shape}")

        self.positions = (rotation_matrix @ (self.positions - rotation_center).T).T + rotation_center
        self.a1s = (rotation_matrix @ self.a1s.T).T
        self.a3s = (rotation_matrix @ self.a3s.T).T
        self._conf._nucleotides[self._index, 9:12] = (rotation_matrix @ self._conf._nucleotides[self._index, 9:12].T).T
        self._conf._nucleotides[self._index, 12:15] = (rotation_matrix @ self._conf._nucleotides[self._index, 12:15].T).T

    @property
    def time(self):
        return self._conf.time

    @property
    def box(self):
        return self._conf.box

    @property
    def energy(self):
        return self._conf.energy

    @property
    def positions(self):
        return self._conf.positions[self._index]

    @positions.setter
    def positions(self, value):
        self._conf._nucleotides[self._index, :3] = value

    @property
    def a1s(self):
        return self._conf.a1s[self._index]

    @a1s.setter
    def a1s(self, value):
        self._conf._nucleotides[self._index, 3:6] = value

    @property
    def a3s(self):
        return self._conf.a3s[self._index]

    @a3s.setter
    def a3s(self, value):
        self._conf._nucleotides[self._index, 6:9] = value

    @property
    def backbone_type(self):
        return self._conf.backbone_type

    @property
    @readonly
    def a2s(self):
        return np.cross(self.a3s, self.a1s)

    @property
    @readonly
    def base_end_positions(self):
        """base end / nucleotide end at base direction"""
        return self.positions + self.a1s * 0.6

    @property
    @readonly
    def base_center_positions(self):
        """base centroid / hydrogen-bonding/repulsion site"""
        return self.positions + self.a1s * 0.4

    @property
    @readonly
    def backbone_center_positions(self):
        """backbone centroid / backbone repulsion site"""
        if self._conf.backbone_type == 'oxDNA2':
            return self.positions - 0.34 * self.a1s + 0.3408 * self.a2s
        elif self._conf.backbone_type == 'RNA':
            return self.positions + self.a1s * -0.4 + self.a3s * 0.2
        else:
            return self.positions + self.a1s * -0.4

    @typing.overload
    def __getitem__(self, sub_index: int) -> Nucleotide:
        ...

    @typing.overload
    def __getitem__(self, sub_index: slice) -> ConfigurationSlice:
        ...

    def __getitem__(self, sub_index: int | slice):
        start, _, step = self._index.indices(len(self._conf._nucleotides))

        if isinstance(sub_index, slice):
            rel_start, rel_stop, rel_step = sub_index.indices(len(self))
            abs_slice = (rel_start * step + start, rel_stop * step + start, rel_step * step)
            return ConfigurationSlice(self._conf, *abs_slice)
        elif isinstance(sub_index, int):
            if sub_index >= len(self) or sub_index < -len(self):
                raise IndexError(f"Index {sub_index} out of bounds for slice of length {len(self)}")
            if sub_index < 0:
                sub_index += len(self)
            return Nucleotide(self._conf, start + step * sub_index)
        else:
            raise ValueError(f"Invalid index type: {type(sub_index)}")

    def __iter__(self):
        for index in range(self._index.start, self._index.stop, self._index.step):
            yield Nucleotide(self._conf, index)

    def __len__(self):
        return max(0, int((self._index.stop - self._index.start) / self._index.step))

    def __repr__(self):
        return f"<ConfigurationSlice time={self.time} slice={self._index} len={len(self)}>"


class Nucleotide:
    def __init__(self, conf: Configuration, index: int):
        self._conf = conf
        self._index = index

    def rotate(self, rotation_matrix: npt.NDArray[np.float64], rotation_center: npt.NDArray[np.float64] | None = None):
        """rotate and modify configuration using rotation matrix"""
        Configuration._check_valid_rotation_matrix(rotation_matrix)

        rotation_center = rotation_center if rotation_center is not None else np.zeros(3)
        if not rotation_center.shape == (3,):
            raise ValueError(f"rotation_center shape must be (3,), not {rotation_center.shape}")

        self.position = (rotation_matrix @ (self.position - rotation_center)) + rotation_center
        self.a1 = rotation_matrix @ self.a1
        self.a3 = rotation_matrix @ self.a3
        self._conf._nucleotides[self._index, 9:12] = rotation_matrix @ self._conf._nucleotides[self._index, 9:12]
        self._conf._nucleotides[self._index, 12:15] = rotation_matrix @ self._conf._nucleotides[self._index, 12:15]

    @property
    def time(self):
        return self._conf.time

    @property
    def box(self):
        return self._conf.box

    @property
    def energy(self):
        return self._conf.energy

    @property
    def position(self) -> npt.NDArray[np.float64]:
        return self._conf.positions[self._index]

    @position.setter
    def position(self, value):
        self._conf._nucleotides[self._index, :3] = value

    @property
    def a1(self) -> npt.NDArray[np.float64]:
        return self._conf.a1s[self._index]

    @a1.setter
    def a1(self, value):
        self._conf._nucleotides[self._index, 3:6] = value

    @property
    def a3(self) -> npt.NDArray[np.float64]:
        return self._conf.a3s[self._index]

    @a3.setter
    def a3(self, value):
        self._conf._nucleotides[self._index, 6:9] = value

    @property
    def backbone_type(self):
        return self._conf.backbone_type

    @property
    @readonly
    def a2(self):
        return np.cross(self.a3, self.a1)

    @property
    @readonly
    def base_end_position(self):
        """base end / nucleotide end at base direction"""
        return self.position + self.a1 * 0.6

    @property
    @readonly
    def base_center_position(self):
        """base centroid / hydrogen-bonding/repulsion site"""
        return self.position + self.a1 * 0.4

    @property
    @readonly
    def backbone_center_position(self):
        """backbone centroid / backbone repulsion site"""
        if self._conf.backbone_type == 'oxDNA2':
            return self.position - 0.34 * self.a1 + 0.3408 * self.a2
        elif self._conf.backbone_type == 'RNA':
            return self.position + self.a1 * -0.4 + self.a3 * 0.2
        else:
            return self.position + self.a1 * -0.4

    def __repr__(self):
        return f"<Nucleotide time={self.time} index={self._index}>"


if typing.TYPE_CHECKING:
    import numpy.typing as npt
