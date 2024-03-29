# traffic_motion_predictor
Given 1 second of traffic and an agent's starting position, predict its trajectory 8 seconds in the future

[The pitch](https://docs.google.com/presentation/d/1E2o_M7UZ1KbnwBISfUV8wHGs3iV3WCgAW15c2zrsciA/edit?usp=sharing)

***
### Setup

#### Setting up Python
1. install python 3.10
2. **(optional)** setup virtual environment

linux/mac:
```sh
python -m venv .env
source .env/bin/activate
```

windows:
```sh
python -m venv .env
./.env/Scripts/activate
```

3. install dependencies
```sh
python -m pip install requirements.txt
```

#### Setting up NuScenes mini dataset

##### Download the Mini Dataset

```
mkdir -p data/sets/nuscenes  # Make the directory to store the nuScenes dataset in.
wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.
tar -xf v1.0-mini.tgz -C data/sets/nuscenes  # Uncompress the nuScenes mini split.
```
(for windows the easiest way is to just run these commands in wsl)

##### Adding the US Map expansion pack
1. Open the [downloads page](https://www.nuscenes.org/download) and go to `Full Dataset (v1.0) > Mini` and click US
2. download US Map expansion pack (v1.3)
3. decompress both packages and drop `basemap/`, `expansion/` and `prediction/` (all from the maps expansion pack) into `v1.0-mini/maps/`

#### Jupyter Notebook


