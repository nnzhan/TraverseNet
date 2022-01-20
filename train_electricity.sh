#!/bin/sh

python main.py --config ./config_electricity/astgcn.json | tee ./log/electricity_astgcn.log
python main.py --config ./config_electricity/dcrnn.json | tee ./log/electricity_dcrnn.log
python main.py --config ./config_electricity/graphwavenet.json | tee ./log/electricity_graphwavenet.log
python main.py --config ./config_electricity/gru.json | tee ./log/electricity_gru.log
python main.py --config ./config_electricity/stgcn.json | tee ./log/electricity_stgcn.log
python main.py --config ./config_electricity/traversenet.json | tee ./log/electricity_traversenet.log
