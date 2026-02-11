from __future__ import annotations
import typing

from .configuration import Configuration, ConfigurationSlice
from .trajectory import Trajectory


class Topology:
    """
    Wrapper for oxDNA topology file

    :param file_path: path to topology file

    Provides list-like interface for accessing strands in a topology file
    Use index or for-in to access strands
    """
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.strands, self.n_monomer = self._parse_topology(f.readlines())

    @staticmethod
    def _parse_topology(lines: list[str]):
        n_monomer, n_strands = map(int, lines[0].split())
        strands: dict[int, list] = {}
        for index, line in enumerate(lines[1:]):
            strand_id, monomer, prev, next = line.split()
            strand_id, prev, next = int(strand_id), int(prev), int(next)
            if prev == -1:
                assert strand_id not in strands
                strands[strand_id] = [index, index, next, [monomer]]
            else:
                assert prev == index - 1
                assert strands[strand_id][1] == prev
                assert strands[strand_id][2] == index
                strands[strand_id][1] = index
                strands[strand_id][2] = next
                strands[strand_id][3].append(monomer)
        assert len(strands) == n_strands
        assert sum(len(strand[3]) for strand in strands.values()) == n_monomer
        assert all(strand[2] == -1 for strand in strands.values())
        return {
            strand_id: Strand(start=strand[0], end=strand[1], sequence=''.join(strand[3]))
            for strand_id, strand in strands.items()
        }, n_monomer

    def __len__(self):
        return len(self.strands)

    def __repr__(self):
        return f'<Topology strands={len(self.strands)} monomers={self.n_monomer}>'

    def __iter__(self):
        return iter(self.strands.values())


class Strand:
    """
    Representation of a strand from oxDNA topology file

    Provides `start`, `end`, `sequence` attributes to access strand details
    """
    def __init__(self, start: int, end: int, sequence: str):
        self.start = start
        self.end = end
        self.sequence = sequence

    @typing.overload
    def slice(self, conf_or_traj: Configuration) -> ConfigurationSlice:
        ...

    @typing.overload
    def slice(self, conf_or_traj: Trajectory) -> TrajectorySlice:
        ...

    def slice(self, conf_or_traj: Configuration | Trajectory):
        if isinstance(conf_or_traj, Configuration):
            return conf_or_traj[self.start:self.end + 1]
        elif isinstance(conf_or_traj, Trajectory):
            return TrajectorySlice(conf_or_traj, self)
        else:
            raise TypeError(f'Expect either Configuration of Trajectory, got {type(conf_or_traj)}')

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return f'<Strand start={self.start} end={self.end}>'


class TrajectorySlice:
    def __init__(self, traj: Trajectory, strand: Strand):
        self._trajectory = traj
        self._strand = strand

    @typing.overload
    def __getitem__(self, index: int) -> ConfigurationSlice:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Generator[ConfigurationSlice, None, None]:
        ...

    def __getitem__(self, index: slice | int):
        if isinstance(index, slice):
            def _slice_iter():
                for conf in self._trajectory[index]:
                    yield self._strand.slice(conf)
            return _slice_iter()
        else:
            return self._strand.slice(self._trajectory[index])

    def __iter__(self):
        yield from self[:]

    def __len__(self):
        return len(self._trajectory)
