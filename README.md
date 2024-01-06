# VisionAlgorithmsForMobileRobotics_MiniProject

# Team Members
1. 
2. 
3. 
4. 


# Tested On
Linux x86 PC  
RAM:  
CPU:   

# Steps to setup
Note: make sure you have miniconda3 installed and working in the terminal  
Note: first navigate into the folder  
```
conda env create -f python_env/conda_mac.yaml
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
You can configure the dataset name inside the main.py  
```
python3 vo_pipeline.py --dataset_name kitti --config config/kitti.yaml
python3 vo_pipeline.py --dataset_name parking --config config/parking.yaml
python3 vo_pipeline.py --dataset_name malaga --config config/malaga.yaml
python3 vo_pipeline.py --dataset_name own --config config/config.yaml
```

Note: the results are saved into a subfolder with the dataset name in the out/ folder 

# Demos
## Demo 1

## Demo 2

## Demo 3

## Demo 4


