#!/bin/sh
# L2 regularization experiments
echo testing MovieLens100K
for var in 1 2 3 4 5 6 7 8 9
do
	echo testing$var
	./LambdaMF -train m.train.weak.10.dat -test m.test.weak.10.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -test_logfile L2log/m.test.10.log$var
	./LambdaMF -train m.train.weak.20.dat -test m.test.weak.20.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -test_logfile L2log/m.test.20.log$var
	./LambdaMF -train m.train.weak.50.dat -test m.test.weak.50.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -test_logfile L2log/m.test.50.log$var
done
var=10
echo testing$var
./LambdaMF -train m.train.weak.10.dat -test m.test.weak.10.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/m.train.10.log$var -test_logfile L2log/m.test.10.log$var -norm L2log/m.norm.10
./LambdaMF -train m.train.weak.20.dat -test m.test.weak.20.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/m.train.20.log$var -test_logfile L2log/m.test.20.log$var -norm L2log/m.norm.20
./LambdaMF -train m.train.weak.50.dat -test m.test.weak.50.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/m.train.50.log$var -test_logfile L2log/m.test.50.log$var -norm L2log/m.norm.50

echo testing Netflix
./LambdaMF -train n.train.weak.10.dat -test n.test.weak.10.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/n.train.10.log$var -test_logfile L2log/n.test.10.log$var -norm L2log/n.norm.10
./LambdaMF -train n.train.weak.20.dat -test n.test.weak.20.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/n.train.20.log$var -test_logfile L2log/n.test.20.log$var -norm L2log/n.norm.20
./LambdaMF -train n.train.weak.50.dat -test n.test.weak.50.dat -L2 0.5 -a 0.0 -e 0.001 -n 250 -train_logfile L2log/n.train.50.log$var -test_logfile L2log/n.test.50.log$var -norm L2log/n.norm.50
