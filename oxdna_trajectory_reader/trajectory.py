from __future__ import annotations
import os
import json
import itertools
import contextlib
import typing

from .configuration import Configuration
from .oxdna_trajectory_reader import read_configurations, read_indicies


CHUNK_SIZE = 20
CACHE_LIMIT_RATIO = 3


class TrajFileIdx:
    def __init__(self, trajectory: TrajReader):
        self.file_path = trajectory.file_path
        self._file_size = trajectory._file_size
        self._chunk_size = trajectory._chunk_size
        self._end_offsets = self._read_idx()

    @property
    def index_file_path(self):
        return f"{self.file_path}.idx"

    @property
    def _is_partial_indicies(self):
        return not self._end_offsets or self._end_offsets[-1] < self._file_size

    def _read_idx(self) -> list[int]:
        with contextlib.suppress(Exception):
            with open(self.index_file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                end_offsets = [int(offset + length) for offset, length, _ in data]
                assert all(i == index for i, (_, _, index) in enumerate(data))
                assert all(s[0] == e for s, e in zip(data[1:], end_offsets[:-1]))
                assert end_offsets[-1] == self._file_size
                return end_offsets
        return []

    def _save_idx(self):
        start_offsets = [0, *self._end_offsets[:-1]]
        lengths = [end - start for start, end in zip(start_offsets, self._end_offsets)]
        indicies = list(zip(start_offsets, lengths, range(len(self._end_offsets))))
        with open(self.index_file_path, "wt", encoding="utf-8") as f:
            json.dump(indicies, f)

    def _get_start_offset(self, index: int):
        if index == 0:
            return 0
        offset = self._end_offsets[index - 1]
        if offset >= self._file_size:
            raise IndexError
        return offset

    def _update_end_offsets(self, first_index: int, offsets: list[int]):
        assert first_index >= 0
        if first_index > len(self._end_offsets):
            raise IndexError(f'first_index={first_index} is not continuous with current indicies {len(self._end_offsets)}')
        if len(self._end_offsets) >= first_index + len(offsets):
            return
        self._end_offsets = self._end_offsets[:first_index] + offsets
        if self._end_offsets[-1] >= self._file_size:
            if len(self._end_offsets) > 1:  # skip configuration file, save only for trajectory file
                self._save_idx()

    def _analyze_offsets(self, target_start_index: int):
        start_index = len(self._end_offsets)
        offsets = read_indicies(self.file_path, self._get_start_offset(start_index),
                                limit=max(self._chunk_size, target_start_index - start_index))
        if not offsets:
            raise ValueError(f'failed to build indicies for "{self.file_path}" from index={start_index}-{target_start_index}')
        self._update_end_offsets(start_index, offsets)

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError(f'invalid index type: {type(index)}')
        if index < 0:
            raise IndexError
        if self._is_partial_indicies and index > len(self._end_offsets):
            self._analyze_offsets(index)
        return self._get_start_offset(index)

    def ensure_indicies(self):
        while self._is_partial_indicies:
            self._analyze_offsets(len(self._end_offsets) + self._chunk_size)

    def get_length(self):
        self.ensure_indicies()
        return len(self._end_offsets)


class TrajReader:
    """
    Wrapper for oxDNA trajectory or configuration file

    :param file_path: path to trajectory or configuration file
    :param chunk_size: number of configurations to read at a time during iteration
    :param backbone: type of `Configuration.backbone`, 'oxDNA1', 'oxDNA2', or 'RNA'

    Provides list-like interface for accessing configurations in a trajectory file
    Use index or for-in to access frames inside a trajectory
    """
    def __init__(self, file_path: str, chunk_size: int = CHUNK_SIZE, backbone_type='oxDNA2'):
        self.file_path = file_path
        self._chunk_size = chunk_size
        self.backbone_type = backbone_type
        self._file_size = os.path.getsize(file_path)
        self._idx = TrajFileIdx(self)
        self._cached_confs: dict[int, Configuration] = {}

    @property
    def length(self):
        return self._idx.get_length()

    def ensure_indicies(self):
        self._idx.ensure_indicies()

    def _load_conf(self, index: int, chunk_size: int):
        assert index >= 0, index
        offset = self._idx[index]
        offsets, configurations = read_configurations(self.file_path, offset, chunk_size)
        self._idx._update_end_offsets(index, offsets)
        self._cached_confs.update({
            index + step: Configuration(time, box, energy, nucleotides, backbone_type=self.backbone_type)
            for step, (time, box, energy, nucleotides) in enumerate(configurations)
        })
        return len(configurations) == chunk_size

    def _get_conf_at(self, index: int, chunk_size: int, forward=True):
        assert index >= 0 and chunk_size > 0, (index, chunk_size)
        if cached := self._cached_confs.pop(index, None):
            return cached
        if forward:
            self._load_conf(index, chunk_size)
        else:
            if not self._load_conf(max(0, index - chunk_size + 1), min(index + 1, chunk_size)):
                raise IndexError
        return self._cached_confs.pop(index)

    def _iter_forward(self, start: int, stop: int | None, step: int):
        assert start >= 0 and (stop or 0) >= 0 and step > 0, (start, stop, step)
        # cache is not helpful when skip is huge
        chunk_size = self._chunk_size if self._chunk_size > step else 1
        for index in itertools.count(start, step):
            if stop is not None and index >= stop:
                break
            try:
                yield self._get_conf_at(index, chunk_size, forward=True)
                while self._cached_confs and (key := next(iter(self._cached_confs))) < index:
                    self._cached_confs.pop(key)
            except IndexError:
                break

    def _iter_backward(self, start: int, stop: int | None, step: int):
        assert start >= -1 and (stop or 0) >= -1 and step < 0, (start, stop, step)
        # cache is not helpful when skip is huge
        chunk_size = self._chunk_size if self._chunk_size > -step else 1
        index = start
        while index > (-1 if stop is None else stop):
            try:
                yield self._get_conf_at(index, chunk_size, forward=False)
                while self._cached_confs and (key := next(iter(self._cached_confs))) > index:
                    self._cached_confs.pop(key)
                index += step
            except IndexError:
                assert not self._idx._is_partial_indicies and index >= self.length
                index = self.length - 1

    @typing.overload
    def __getitem__(self, index: int) -> Configuration:
        ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Generator[Configuration, None, None]:
        ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            if index < 0:
                if index < -self.length:
                    raise IndexError(f'index={index} is out of bounds for trajectory length={self.length}')
                index += self.length
            while len(self._cached_confs) > CACHE_LIMIT_RATIO * self._chunk_size:
                self._cached_confs.pop(next(iter(self._cached_confs)))
            return self._get_conf_at(index, self._chunk_size)
        elif isinstance(index, slice):
            if index.step == 0:
                raise ValueError('slice step cannot be zero')
            if (step := int(index.step or 1)) > 0:
                if (index.start or 0) < 0 or (index.stop or 0) < 0:
                    return self._iter_forward(*index.indices(self.length))
                return self._iter_forward(index.start or 0, index.stop, step)
            else:
                if index.start is None or index.start < 0 or (index.stop or 0) < 0:
                    return self._iter_backward(*index.indices(self.length))
                return self._iter_backward(index.start, index.stop, step)
        else:
            raise TypeError(f'invalid index type: {type(index)}')

    def __iter__(self):
        return self._iter_forward(0, None, 1)

    def __len__(self):
        return self.length
