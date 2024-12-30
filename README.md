# oxdna_trajectory_reader

This is a module for reading oxDNA trajectory data at faster speed.


## Usage

```python
from oxdna_trajectory_reader import Trajectory, dumps_configurations

trajectory = Trajectory('trajectory.dat')
for configuration in trajectory:
    print(configuration.time)
    print(configuration.box)
    print(configuration.energy)
    print(configuration.positions[:10])
    print(configuration.a1s[:10])
    print(configuration.a3s[:10])

    nucleotide = configuration[0]
    nucleotides_slice = configuration[3:10]

third_conf_frame = trajectory[2]

with open('traj_slice.dat', 'wt') as f:
    f.write('\n'.join(dumps_configurations([traj[i] for i in range(5, 10)])))
```
