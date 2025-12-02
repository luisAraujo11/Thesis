#!/bin/bash

study="$1"

bash runMrisPreproc.sh "$study"
bash runGLMs.sh "$study"
bash runClustSims.sh "$study"
