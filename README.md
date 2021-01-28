# Crocodile

## Arduino Libraries

The hardware requires the following libraries:

  - [Biodata](https://github.com/eringee/Biodata)
  - [Chrono](https://github.com/SofaPirate/Chrono)
  - [Circular Buffer](https://github.com/rlogiacco/CircularBuffer)
  - [Liquid Crystal I2C](https://www.arduino.cc/reference/en/libraries/liquidcrystal-i2c/)
 

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
