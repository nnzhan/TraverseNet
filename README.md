# Instructions

## Preparation

1. Download data from https://github.com/Davidham3/STSGCN
2. Make the data folder and move the downloaded dataset into the data folder
3. Pre-process data:
```
python proc_data.py
```

## Training
For our model
```
python main.py --config ./config/traversenet.json
```
For baseline models
```
python main.py --config ./config/astgcn.json
python main.py --config ./config/dcrnn.json
python main.py --config ./config/graphwavenet.json
python main.py --config ./config/gru.json
python main.py --config ./config/stgcn.json
```
