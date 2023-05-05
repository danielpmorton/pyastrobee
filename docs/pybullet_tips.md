#  Assorted Pybullet notes to remember

- Pybullet's timestep (via `setTimeStep`) is left at the default value of 240 Hz (`dt = 1/240`). The quickstart guide gives some reasons why this should be left at this value, but the most important implication of this is that it dictates how velocities are defined! If we calculate velocities assuming a `dt` of anything other than `1/240` seconds, the values will be off!
