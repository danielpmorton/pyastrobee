# Turn the xacro from the astrobee repo into a urdf we can load into Bullet
# The parameters here can be changed if desired - see the NASA tutorials
rosrun xacro xacro world:=iss top_aft:=perching_arm bot_aft:=empty bot_front:=empty ns:=honey prefix:=honey pyastrobee/resources/urdf/astrobee.xacro > pyastrobee/resources/urdf/astrobee.urdf
