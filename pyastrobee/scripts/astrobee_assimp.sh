#!/bin/bash

# TODO:
# Directories should probably be command-line arguments
# Should check and throw a warning if the file is not found

MESH_DIR=$HOME/pyastrobee/pyastrobee/assets/meshes/astrobee/dae # (Change as needed)

# Run assimp on each dae file in the directory, and output the OBJ to its own folder
pushd $MESH_DIR
for file in *.dae
do
    assimp export $file "../obj/${file%.*}.obj"
done
popd
