#!/bin/sh

python main.py --config ./config_solar/astgcn.json | tee ./log/solar_astgcn.log
python main.py --config ./config_solar/dcrnn.json | tee ./log/solar_dcrnn.log
python main.py --config ./config_solar/graphwavenet.json | tee ./log/solar_graphwavenet.log
python main.py --config ./config_solar/gru.json | tee ./log/solar_gru.log
python main.py --config ./config_solar/stgcn.json | tee ./log/solar_stgcn.log
python main.py --config ./config_solar/traversenet.json | tee ./log/solar_traversenet.log
