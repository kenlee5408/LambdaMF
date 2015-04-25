#!/bin/sh
for var in 1 2 3 4 5 6 7 8 9 10
do
	echo testing$var
	./LambdaMF -train m.train.weak.10.dat -test m.test.weak.10.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/m.test.10.log$var
	./LambdaMF -train m.train.weak.20.dat -test m.test.weak.20.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/m.test.20.log$var
	./LambdaMF -train m.train.weak.50.dat -test m.test.weak.50.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/m.test.50.log$var
done
./LambdaMF -train n.train.weak.10.dat -test n.test.weak.10.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/n.test.10.log$var
./LambdaMF -train n.train.weak.20.dat -test n.test.weak.20.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/n.test.20.log$var
./LambdaMF -train n.train.weak.50.dat -test n.test.weak.50.dat -a 0.5 -e 0.001 -n 250 -test_logfile log/n.test.50.log$var
