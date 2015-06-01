#!/bin/sh
g++ LambdaMF_test_converge.cpp -O3 -o testConverge
./testConverge -train m.train.weak.10.dat -test m.test.weak.10.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/m.test.10.conv -train_logfile log/m.train.10.conv
./testConverge -train m.train.weak.20.dat -test m.test.weak.20.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/m.test.20.conv -train_logfile log/m.train.20.conv
./testConverge -train m.train.weak.50.dat -test m.test.weak.50.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/m.test.50.conv -train_logfile log/m.train.50.conv
./testConverge -train n.train.weak.10.dat -test n.test.weak.10.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/n.test.10.conv -train_logfile log/n.train.10.conv
./testConverge -train n.train.weak.20.dat -test n.test.weak.20.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/n.test.20.conv -train_logfile log/n.train.20.conv
./testConverge -train n.train.weak.50.dat -test n.test.weak.50.dat -e 0.5 -g 0.001 -n 250 -test_logfile log/n.test.50.conv -train_logfile log/n.train.50.conv
