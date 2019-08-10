# Crocodile


## Dataset

You first need to extract the frame from the videos by running:
`python dataset.py VIDEO_PATH DATASET_PATH [-r RESOLUTION]`

- VIDEO_PATH: is the path to the video you want to extract the frame from.
- DATASET_PATH: is the path where you want to save the extracted frames.
- RESOLUTION (Optional): If you precise a resolution, this will also resize the images to the appropriate resolution.

## To run the code:

`python train.py -r RESOLUTION --output-path OUTPUT_PATH --path_to_dataset DATASET_PATH`

- OUTPUT_PATH: This is the path where you want to save the results. It will create
a sub-folder `img` where the samples are saved for each epochs.
- DATASET_PATH: This is the link where you want to save the dataset.
- RESOLUTION: This is the resolution at which you want to generate images.
- There is a bunch of other options you can play with. To see a list of all options just run `python train.py --help`
