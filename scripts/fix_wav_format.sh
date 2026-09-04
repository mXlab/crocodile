#!/usr/bin/env bash

if [ "$#" -lt "2" ]; then
  echo "Usage: $0 [input_wav_file] [output_wav_file] [n_channels=4] [samplerate=1000]"
  exit
fi

input_wav_file="$1"
output_wav_file="$2"

if [ "$#" -ge "3" ]; then
  n_channels="$3"
else
  n_channels=4
fi

if [ "$#" -ge "4" ]; then
  samplerate="$4"
else
  samplerate=1000
fi


cat $input_wav_file | out123 -r $samplerate -c $n_channels  -w $output_wav_file
