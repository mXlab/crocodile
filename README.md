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
1. `screen -S mlflow_server`  or `screen -r mlflow_server` if the screen is already created
2. `sh scripts/start_mlflow_server.sh`

## Training a model:
To train a model run the following commands:
1. `module load gcc arrow python`
2. `source .env/bin/activate`
3. `python scripts/train.py`

## Example on how to use a Model:
```python
from crocodile.trainer import load_generator
from FastGAN import FastGANConfig
config = FastGANConfig()
generator = load_generator(config)
noise = torch.tensor(1, generator.latent_dim)
img = generator.generate()
```
