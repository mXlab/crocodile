# Crocodile

## Arduino Libraries

The hardware requires the following libraries:

  - [Biodata](https://github.com/eringee/Biodata)
  - [Chrono](https://github.com/SofaPirate/Chrono)
  - [Circular Buffer](https://github.com/rlogiacco/CircularBuffer)
  - [Liquid Crystal I2C](https://www.arduino.cc/reference/en/libraries/liquidcrystal-i2c/)
 
In addition to the Teensyduino add-on to program the Teensy 3.2 microcontroller in Arduino IDE.  INSTALL ALL LIBRAIRIES when prompted.
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
```conda create -f conda/crocodile.yml```
3. To activate:  
```conda activate crocodile```
4. To deactivate:
```conda deactivate```



--------------------------
# New version

## Setting up on Beluga
### On the login node:
1. Clone repository: `git clone --recurse-submodules git@github.com:a3lab/crocodile.git`
2. `cd crocodile`
3. Loading python: `module load python`
4. Create dir to store packages: `mkdir packages`
5. Download packages: `pip download --no-deps -r requirements/downloads.txt -d packages`
6. Create a virtualenv: `virtualenv --no-download .env`
7. Activate environement: `source .env/bin/activate`
8. Upgrade pip: `pip install --no-index --upgrade pip`
9. Install all dependencies: `pip install --no-index -r requirements/beluga.txt`
10. Install crocodile package: `pip install -e . --no-deps`



### Once on a compute node:
1. Create a virtualenv: `virtualenv --no-download .env`
2. Activate environement: `source .env/bin/activate`
3. Upgrade pip: `pip install --no-index --upgrade pip`
4. Install all dependencies: `pip install --no-index -r requirements/beluga.txt`
5. Install crocodile package: `pip install -e . --no-deps`

## Running job on Beluga
### Ask for an interactive Job
`salloc --time=1:0:0 --mem-per-cpu=3G --ntasks=2`

### Run training
``

