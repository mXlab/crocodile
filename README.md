# Crocodile

## Requirements

Requires python 3.11 or higher.

## Arduino Libraries

The hardware requires the following libraries:

- [Biodata](https://github.com/eringee/Biodata)
- [Chrono](https://github.com/SofaPirate/Chrono)
- [Circular Buffer](https://github.com/rlogiacco/CircularBuffer)
- [Liquid Crystal I2C](https://www.arduino.cc/reference/en/libraries/liquidcrystal-i2c/)

In addition to the Teensyduino add-on to program the Teensy 3.2 microcontroller in Arduino IDE. INSTALL ALL LIBRAIRIES when prompted.

- [Teensyduino](https://www.pjrc.com/teensy/teensyduino.html)

## Installation

To install the pygan library run:

```
git clone --recurse-submodules git@github.com:a3lab/crocodile.git
cd crocodile/
pip install -e .
cd pygan/face3d/3DDFA_V2
sh ./build.sh
```

## Dataset

You first need to extract the frame from the videos by running:
`python dataset.py VIDEO_PATH DATASET_PATH [-r RESOLUTION]`

- VIDEO_PATH: is the path to the video you want to extract the frame from.
- DATASET_PATH: is the path where you want to save the extracted frames.
- RESOLUTION (Optional): If you precise a resolution, this will also resize the images to the appropriate resolution.

## Training a unconditional GAN:

`python train.py -r RESOLUTION --output-path OUTPUT_PATH --path-to-dataset DATASET_PATH`

- OUTPUT_PATH: This is the path where you want to save the results. It will create
  a sub-folder `img` where the samples are saved for each epochs.
- DATASET_PATH: This is the link where you want to save the dataset.
- RESOLUTION: This is the resolution at which you want to generate images.
- There is a bunch of other options you can play with. To see a list of all options just run `python train.py --help`

## Training a GAN conditioned on the labels:

`python train_conditional.py OUTPUT_PATH -r RESOLUTION --path-to-dataset DATASET_PATH`

- Same option as command above

## Training a GAN conditioned on the biodata:

`python train_with_biodata.py OUTPUT_PATH -r RESOLUTION --path-to-dataset DATASET_PATH --path-to-biodata PATH_TO_BIODATA`

- PATH_TO_BIODATA: This the path to the csv file containing the biodata, if not specified automatically look for the file "LaurenceHBS-Nov919mins1000Hz-Heart+GSR-2channels.csv" in the folder DATASET_PATH.

## Setting up the environment

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create conda environment:  
   `conda create -f conda/crocodile.yml`
3. To activate:  
   `conda activate crocodile`
4. To deactivate:
   `conda deactivate`

---

# New version

## Installation
Run the following command to install the package:
```bash
sh scripts/install.sh
```

## Starting the mlflow server:
To start the mlflow server run the following commands:
1. `screen -S mlflow_server`  or `screen -r mlflow_server` if the screen is already created
2. `sh scripts/start_mlflow_server.sh`

## Training a model:
To train a model run the following command:
```bash
python crocodile/train.py
```
