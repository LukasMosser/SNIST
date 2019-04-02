#!/bin/sh
# file : /data_generation/generate_amplitudes.sh
#
cd ./forward_model/devito/
docker-compose run generate-train
docker-compose run generate-test 