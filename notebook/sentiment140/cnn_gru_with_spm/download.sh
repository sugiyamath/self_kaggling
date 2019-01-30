#!/bin/bash
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip
mv training.1600000.processed.noemoticon.csv train.csv
mv testdata.manual.2009.06.14.csv test.csv
