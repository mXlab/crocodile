# BioData Feature Extraction (Arduino)

Real-time physiological data collection and feature extraction using Arduino/Teensy microcontrollers.

## Overview

This system reads raw heart rate, EDA (skin conductance), and respiration signals from wearable sensors and extracts real-time features on-device. It can run on actual hardware or be emulated on a computer using EpoxyDuino.

## Hardware

- **Microcontroller**: Teensy 3.2 (or compatible Arduino board)
- **Sensors**: Photoplethysmograph (heart), electrodes (EDA), thermistor (respiration)
- **Sample rate**: 100 Hz (simulated time)

## Source Code

- `src/feature_extract.ino` -- Main sketch: reads CSV sensor data, extracts 20 real-time features, outputs results to CSV
- `src/CsvReader.h` / `src/CsvWriter.h` -- CSV I/O utilities for Arduino

## Extracted Features (20 total)

- **Heart (4)**: BPM, signal amplitude, peak detection metrics
- **EDA (2)**: Skin conductance level, response metrics
- **Respiration (14)**: Breathing rate, depth, variability, sigh detection

## Libraries (Git Submodules)

| Library | Description |
|---|---|
| [BioData](libraries/BioData/) | Heart, EDA, and respiration sensor processing (by Erin Gee) |
| [Plaquette](libraries/Plaquette/) | Object-oriented real-time signal processing |
| [EpoxyDuino](libraries/EpoxyDuino/) | Arduino emulation framework for Linux/Mac testing |
| [CSV Parser](libraries/CSV%20Parser/) | CSV reading/writing for Arduino |

Each library has its own README with detailed documentation.

## Setup

Clone with submodules:

```bash
git submodule update --init --recursive
```

### On hardware

Open `src/feature_extract.ino` in the Arduino IDE or PlatformIO. Upload to your Teensy/Arduino board.

### Emulated (desktop)

Using EpoxyDuino, you can compile and run the sketch on Linux or macOS without hardware. See the [EpoxyDuino README](libraries/EpoxyDuino/README.md) for build instructions.
