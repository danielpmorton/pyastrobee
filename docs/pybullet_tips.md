#  Assorted Pybullet notes to remember

### Timesteps
- Pybullet's timestep (via `setTimeStep`) is currently set to 350 Hz (`dt = 1 / 350`) (Check our `initialize_pybullet()` function to confirm this in case it changed). Pybullet's default timestep is 240 Hz, and the quickstart guide gives some reasons why this should be left at 240 Hz (based on solver iterations and error reduction parameters, which we might need to experiment with), but we increased this value because it improved the stability of deformable objects. 
- Importantly, this timestep dictates how velocities are defined. If we calculate velocities assuming a `dt` of anything other than our current timestep (`pybullet.getPhysicsEngineParameters()["fixedTimeStep"]`), the values will be off!
- Note that if you just call `pybullet.connect()`, the timestep will default to 240 Hz. So, `initialize_pybullet` is the better way to approach this to avoid bugs. 
