from __future__ import annotations
import numpy as np
import numpy.typing as npt


def read_configurations(file_path: str, offset: int, limit: int) -> tuple[list[int], list[tuple[
    int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]
]:
    """
    Read up to number of `limit` configurations from trajectory, starting from file cursor `offset`

    :param file_path: Path to trajectory file
    :param offset: Start reading from this file cursor offset
    :param limit: Read up to this number of configurations, or until end of file
    :return: Tuple of list of cursor offsets at end of each configuration, and list of configurations,
        each a tuple of
        - Time
        - Box dimensions
        - Energy
        - Nucleotide vectors
    """
    ...


def read_indicies(file_path: str, offset: int, limit: int) -> list[int]:
    """
    Read cursor offsets at end of each configuration, useful for building trajectory indicies

    :param file_path: Path to trajectory file
    :param offset: Start reading from this file cursor offset
    :param limit: Read up to this number of configurations, or until end of file
    :return: List of cursor offsets
    """
    ...


def dumps_configurations(
    configurations: list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]
) -> list[str]:
    """
    Serialize configurations to strings using trajectory file format

    :param configurations: List of tuples of
        - time: int
        - box: np.array
        - energy: np.array
        - nucleotides: np.array
    :return: List of string per configuration
    """
    ...
