#!/bin/bash

# TODO:
# Directories should probably be command-line arguments
# Should check and throw a warning if the file is not found
# Delete unnecessary files generated from the process

# (Change the directory / locations of the compiled programs as needed)
export OBJ_DIR=$HOME/pyastrobee/pyastrobee/assets/meshes/iss/obj
export VHACD=$HOME/software/v-hacd/app/build/TestVHACD
export OBJ2SDF=$HOME/software/bullet3/build_cmake/Extras/obj2sdf/App_obj2sdf

# Go through each directory, find the associated obj file (example: cupola/cupola.obj)
# then run VHACD and OBJ2SDF on the file
pushd $OBJ_DIR
for dir in $OBJ_DIR/*/
do
    pushd $dir
    name=${PWD##*/}
    # Can also use the _minimal version of the objs to neglect lights/handles
    filename="${name}.obj" 
    $VHACD $filename -e 0.01 -d 15 -r 200000
    $OBJ2SDF --fileName="$filename"
    popd
done
popd
