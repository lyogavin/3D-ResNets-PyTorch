#!/bin/bash

#python -m util_scripts.kinetics_json ./kinetics/ 700 ./HMDB-51/eat jpg ./HMDB-51/eat/anno.json
python -m util_scripts.hmdb51_json ./data/HMDB-51/testTrainMulti_7030_splits/ 700 ./HMDB-51/eat jpg ./HMDB-51/eat/anno.json

