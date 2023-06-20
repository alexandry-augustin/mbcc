#!/bin/bash

#DIR="../bin/Release/"
DIR=$1 #FIXME needs to be absolute path
python run_aggregation.py $DIR/ResultsMIBCC "" ""
python run_aggregation.py $DIR/stats/ TrueLabel accuracy
python run_aggregation.py $DIR/stats/ ConfusionMatrix cm_error
