from .configuration import Configuration, ConfigurationSlice


class Topology:
    def __init__(self, filename: str):
        with open(filename, 'r') as f:
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
    def __init__(self, start: int, end: int, sequence: str):
        self.start = start
        self.end = end
        self.sequence = sequence

    def slice(self, conf: Configuration) -> ConfigurationSlice:
        return conf[self.start:self.end + 1]

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        return f'<Strand start={self.start} end={self.end}>'
