# Vision Algorithms for Mobile Robotics (VAMR) Mini project

# Team Members
1. Hardik Shah
2. Deepak Ganesh
3. Aniruddha Sundararajan
4. Deepana Ishtaweera

# Tested On
Linux x86 PC  
RAM:  
CPU:   

# Steps to setup
Note: make sure you have miniconda3/ anaconda installed and working in the terminal  
Note: first navigate into the folder  
```
conda env create -f python_env/conda_config.yaml
```

# Download the datasets
```
mkdir data && cd data
wget -O parking.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/parking.zip
unzip parking.zip
wget -O kitti05.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/kitti05.zip
unzip kitti05.zip
wget -O malaga.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/malaga-urban-dataset-extract-07.zip
unzip malaga.zip
mv malaga-urban-dataset-extract-07 malaga
```

# Run the app
Use the following python commands to run the vo pipeline for different datasets. Make sure to navigate to the folder root before running the commands.
```
python3 vo_pipeline.py --dataset_name kitti --config config/kitti.yaml
python3 vo_pipeline.py --dataset_name parking --config config/parking.yaml
python3 vo_pipeline.py --dataset_name malaga --config config/malaga.yaml
python3 vo_pipeline.py --dataset_name own --config config/custom.yaml
```

Note: the results are saved into a subfolder with the dataset name in the out/ folder 

# Screencasts of the Datasets
## Kitti
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/7yigfi7p3LI/0.jpg)](http://www.youtube.com/watch?v=7yigfi7p3LI)

## Malaga
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/fzCBMkicLZY/0.jpg)](http://www.youtube.com/watch?v=fzCBMkicLZY)

## Parking
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/i9yv4T7ghO8/0.jpg)](http://www.youtube.com/watch?v=i9yv4T7ghO8)

## Custom Dataset
### VO Result
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/Ynx1Ka45ENs/0.jpg)](http://www.youtube.com/watch?v=Ynx1Ka45ENs)

### RAW Video
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/abxF24q7mJU/0.jpg)](http://www.youtube.com/watch?v=abxF24q7mJU)

### Calibration Video
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/8JlmjzX1FCE/0.jpg)](http://www.youtube.com/watch?v=8JlmjzX1FCE)

# File Structure
```
├── config
│   ├── kitti.yaml
│   ├── malaga.yaml
│   ├── parking.yaml
│   ├── custom.yaml
├── data
│   ├── kitti
│   │   ├──...
│   ├── malaga
│   │   ├──...
│   ├── parking
│   │   ├──...
│   ├── custom
│   │   ├──...
├── out
│   ├── kitti
│   │   ├──...
│   ├── malaga
│   │   ├──...
│   ├── parking
│   │   ├──...
│   ├── custom
│   │   ├──...
├── python_env
│   ├── conda_mac.yml
│   ├── conda_ubuntu_x64.yml
│   └── requirements.txt
|── create_video.py
|── data_loader.py
├── dev.py
├── estimate_campose.py
├── feature_extractor.py
├── frame_state.py
├── klt_tracker.py
├── visualizer.py
├── vo_pipeline.py
└── vo_project_statement.pdf
```