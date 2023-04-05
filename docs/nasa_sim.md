# Notes on NASA's simulation environment

## Build/Install

The go-to link here is the [Non-NASA Astrobee Install Guide](https://nasa.github.io/astrobee/v/develop/install-nonNASA.html), which contains pretty much everything you need to set up their code. Unfortunately, there are a lot of random bugs you might encounter along the way

When installing this, you'll want to be on a machine with:
- Ubuntu 20.04
- No prior install of OpenCV
- No prior install of ROS
- No Stanford AFS software
- Sudo and root access
- ... Basically, as close as you can get to a completely "blank-slate" machine

Ros Noetic will be installed during this process. However, let their build scripts install ROS -- don't install it ahead of time!

After the step where you clone their repository, you may want to `cd $ASTROBEE_WS/src` and `git checkout develop`, which has the most up-to-date code. I had to do this, but NASA updated their `master` branch and this may no longer be needed.

They mention this very briefly, but make sure you `sudo apt-get update` and `sudp apt-get upgrade` (this is quite important and easy to miss)

If everything goes well during the "Install Dependencies" step (where most of the bugs are encountered), you'll get to the "Building the Code" section. If cmake fails at this step, try re-building it again with a `-j1` flag. This will take longer to build, but it is more reliable on low-memory machines.

To test if this install worked, run the following:
```
cd $HOME/astrobee # Or ASTROBEE_WS, if installed in a different location
source devel/setup.bash
roslaunch astrobee sim.launch dds:=false robot:=sim_pub rviz:=true sviz:=true
```
This should bring up both a Gazebo and an RVIZ window displaying the astrobee inside the ISS

If this stil isn't working:
- Check that you don't have any virtual environments currently active
- Try anothere re-build
- Delete the repo and start over, paying *really* close attention to any warnings that pop up in the terminal (some dependencies might not actually get installed when they should have, and then you have to manually install them with `apt`)
- (nuclear option 💣) Wipe the computer and start over (unfortunately, not joking)

### Docker

Alternatively, you can use Docker. This means there won't be a local build of the code on the machine, but you can probably get away with this if you're not using ROS much (and this repo doesn't rely on it currently). If this is the case, check out the following resources:
- [Install Docker Enfine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Linux post-installation steps for Docker Engine](https://docs.docker.com/engine/install/linux-postinstall/)
- [nasa/astrobee Installation page](https://github.com/nasa/astrobee/blob/master/INSTALL.md) (see the Docker option)

The Docker build worked on the first try for us, which was a pleasant surprise.

To run Gazebo/RViz with Docker, include the args as follows: `./run.sh --remote --args "rviz:=true sviz:=true"` (where `run.sh` is inside the `docker/` directory). 

Originally, Docker only worked on the `develop` branch, but that bug should be fixed now. 

## Using the simulator

First, start up the simulator
```
cd $HOME/astrobee # Or ASTROBEE_WS, if installed in a different location
source devel/setup.bash
roslaunch astrobee sim.launch dds:=false robot:=sim_pub rviz:=true sviz:=true
```

There may be some errors that show up at the start of the program from the `graph_localizer` and the `imu_integration`, but I found that these can be safely ignored and it seems to work fine. 

If the robot is not visible but all of the frames are, un-check and then re-check the Debug → / checkbox under the Rviz Displays section in the bottom left corner. 

If the robot is randomly floating around the ISS upoin starting the simulation, this is not good (the localizer is totally busted). I've only seen this when starting the simulation and the processes in two separate terminal windows. The `roslaunch` one-liner in the code block above should hopefully work.

Interacting with the simulation is via the `teleop_tool`, for example:
```
rosrun executive teleop_tool -undock
rosrun executive teleop_tool -get_pose
rosrun executive teleop_tool -get_state
rosrun executive teleop_tool -move -relative -pos "1 2 0.5"
```

We've only been using a few flags like `rviz` and `sviz` so far, but there are more you can include on launch. If not specified, enable with `flag:=true`. See the [Running the Sim](https://nasa.github.io/astrobee/v/develop/running-the-sim.html) page for more info
- `pose`: Starting pose. For example, `pose:="x y z qx qy qz qw"`
- `gds`: Starts the Ground Data System
- `rviz`: Starts RVIZ
- `sviz`: Starts Gazebo
- `gviz`: Starts the GNC visualizer
- `dds`: Starts communication nodes
- `speed`: Simulation speed multiplier (1 = real time)
- `ns`: Namespace (for using multiple robots)
- `robot`: Which robot config file to use (leave this as sim_pub for now)
- `default_robot`: If you want to launch the world without a robot, set this `false`
- `perch`: Starts astrobee in a perch-ready position
- `world`: "iss" (default) or "granite"
- `debug`: node name to debug, For example, `"executive"`

The GNC visualizer (see the `gviz` flag above) is a cool way to see some of the localization info, but it's super slow and crashes a lot. 

The `gds` flag will not work until you have the RTI libraries / communication nodes directly from NASA. 
