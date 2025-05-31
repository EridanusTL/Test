#!/bin/bash

rm -r install

cmake --fresh -B build -S .
time make install -j --no-print-directory -C build