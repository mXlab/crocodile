# Crocodile

## Arduino Libraries

The hardware requires the following libraries:

- [Biodata](https://github.com/eringee/Biodata)
- [Chrono](https://github.com/SofaPirate/Chrono)
- [Circular Buffer](https://github.com/rlogiacco/CircularBuffer)
- [Liquid Crystal I2C](https://www.arduino.cc/reference/en/libraries/liquidcrystal-i2c/)

In addition to the Teensyduino add-on to program the Teensy 3.2 microcontroller in Arduino IDE. INSTALL ALL LIBRAIRIES when prompted.

- [Teensyduino](https://www.pjrc.com/teensy/teensyduino.html)

## Installation

Run the following command to install the package:

```bash
sh scripts/install.sh
```

## Starting the mlflow server:

To start the mlflow server run the following commands:

1. `screen -S mlflow_server` or `screen -r mlflow_server` if the screen is already created
2. `sh scripts/start_mlflow_server.sh`

## Training a model:

To train a model run the following commands:

1. `module load gcc arrow python`
2. `source .env/bin/activate`
3. `python scripts/train.py`

## Example on how to load a Model:
Models are available at: https://drive.google.com/drive/folders/1OjW0I-9Ht8ql98YiRBb4Tum0l_B3I67a?usp=sharing

### Loading a generator:
```python
import torch
from crocodile.loader import load_from_path
generator = load_from_path(path)
noise = generator.noise(num_samples)
img = generator.generate(noise)
```

### Loading an encoder:
```python
import torch
from crocodile.encoder import Encoder
encoder = Encoder.load(path)
latent = encoder(biodata)
```

## Deploying MLFlow server on Google Cloud

1. `gcloud run deploy --project crocodile-333216 --add-cloudsql-instances=crocodile-mlflow --memory 1Gi`
2. Connect to : https://crocodile-gqhfy6c73a-uc.a.run.app/
