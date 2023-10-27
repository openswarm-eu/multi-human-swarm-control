#!/bin/bash

# Script to run commands during the webviz user study.
# 
# 1) To run the docker container with the ARGoS simulation:
#
#   $ ./experiment.sh run <condition> <order>
#
#    where 'condition' = { direct | indirect | debug },
#          'order' = { 1 | 2 }
#
# 2) To compress the result data:
#
#   $ ./experiment.sh compress
#

mode=$1
condition=$2
order=$3

echo "Mode : $mode"
echo "Order: $order"

if [ $mode = "run" ]
then
    docker rm -f argos
    echo "Starting docker container..."
    cd ~/Documents/hsi-experiment
    docker run --name argos -it --network="host" -v $(pwd)/results:/home/docker/multi-human-swarm-control/results -w /home/docker/multi-human-swarm-control genki15/multi-human-swarm-control python3 src/web_app/app.py -m $condition -o $order
elif [ $mode = "compress" ]
then
    echo "Compressing experiment data..."
    cd ~/Documents/hsi-experiment
    filename=$(date +"results_%Y-%m-%d_%H-%M")
    mv results $filename
    zip -r ./$filename.zip $filename/*
else
    echo "Unrecognized mode"
fi
