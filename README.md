# Vision Algorithms for Mobile Robotics (VAMR) Mini project

Report can be found [here]()

# Team Members
1. Hardik Shah (hashah@ethz.ch)
2. Deepak Ganesh (dganesh@ethz.ch)
3. Aniruddha Sundararajan (asundararaja@ethz.ch)
4. Deepana Ishtaweera (dishtaweera@ethz.ch)

# Tested on
Screencasts were recorded on, 
ROG Zephyrus G15 GA503 GA503QM-HQ121R    
OS: Ubuntu 22.04  
CPU: 3.0 GHz AMD Ryzen 9 5900HS  
RAM: 16 GB 3200MHz  

# Steps to setup
## Setting up conda environment
Note: make sure you have miniconda3/ anaconda installed and working in the terminal  
Note: first navigate into the folder  
```
conda env create -f python_env/conda_config.yml
```
## Optional: using pyenv virtual environment
```
pip3 install -r python_env/requirements.txt
```

# Download the datasets
Use the following commands to download the benchmark datasets.
```
mkdir data && cd data
wget -O parking.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/parking.zip
unzip parking.zip
wget -O kitti05.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/kitti05.zip
unzip kitti05.zip
wget -O malaga.zip https://rpg.ifi.uzh.ch/docs/teaching/2023/malaga-urban-dataset-extract-07.zip
unzip malaga.zip
mv malaga-urban-dataset-extract-07 malaga
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12IQMiJbkg5LW9epJfGxKL8U6VYO33fu7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12IQMiJbkg5LW9epJfGxKL8U6VYO33fu7" -O own.zip && rm -rf /tmp/cookies.txt
unzip own.zip
```

The data folder structure should be as follows.
```
├── data
│   ├── kitti
│   │   ├──05
│   │   │   ├── image_0
│   │   │   │   ├── ...
│   │   │   ├── image_1
│   │   │   │   ├── ...
│   │   │   ├── calib.txt
│   │   │   ├── times.txt
│   │   ├──poses
│   │   │   ├── ...
│   ├── malaga
│   │   ├── ...
│   ├── parking
│   │   ├──images
│   │   ├──K.txt
│   │   ├──poses.txt
│   ├── own
│   │   ├── ...
```

# Run the app
Usage of the vo_pipeline.py file
```
usage: vo_pipeline.py [-h] [--dataset_dir DATASET_DIR] [--dataset_name DATASET_NAME] [--config CONFIG]

Visual Odometry Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --dataset_dir DATASET_DIR
                        Path to the dataset directory
  --dataset_name DATASET_NAME
                        Name of the dataset: can be kitti, malaga, parking or own
  --config CONFIG       Path to the config file: can be config/params.yaml, config/params_kitti.yaml, config/params_malaga.yaml,
                        config/params_parking.yaml or config/params_own.yaml
```

Use the following python commands to run the vo pipeline for different datasets. Make sure to navigate to the folder root before running the commands.
```
python3 vo_pipeline.py --dataset_name kitti --config config/kitti.yaml
python3 vo_pipeline.py --dataset_name parking --config config/parking.yaml
python3 vo_pipeline.py --dataset_name malaga --config config/malaga.yaml
python3 vo_pipeline.py --dataset_name own --config config/own.yaml
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

