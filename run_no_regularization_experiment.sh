#!/bin/sh
# NO regularization experiments
echo testing MovieLens100K
for var in 1 2 3 4 5 6 7 8 9
do
	echo testing$var
	./LambdaMF -D 100 -train m.train.weak.10.dat -test m.test.weak.10.dat -a 0.0 -e 0.001 -n 250 -test_logfile NOlog/m.test.10.log$var -VERBOSE 1
	./LambdaMF -D 100 -train m.train.weak.20.dat -test m.test.weak.20.dat -a 0.0 -e 0.001 -n 250 -test_logfile NOlog/m.test.20.log$var -VERBOSE 1
	./LambdaMF -D 100 -train m.train.weak.50.dat -test m.test.weak.50.dat -a 0.0 -e 0.001 -n 250 -test_logfile NOlog/m.test.50.log$var -VERBOSE 1
done
var=10
echo testing$var
./LambdaMF -D 100 -train m.train.weak.10.dat -test m.test.weak.10.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/m.train.10.log$var -test_logfile NOlog/m.test.10.log$var -norm NOlog/m.norm.10 -VERBOSE 1
./LambdaMF -D 100 -train m.train.weak.20.dat -test m.test.weak.20.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/m.train.20.log$var -test_logfile NOlog/m.test.20.log$var -norm NOlog/m.norm.20 -VERBOSE 1
./LambdaMF -D 100 -train m.train.weak.50.dat -test m.test.weak.50.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/m.train.50.log$var -test_logfile NOlog/m.test.50.log$var -norm NOlog/m.norm.50 -VERBOSE 1

echo testing Netflix
./LambdaMF -D 100 -train n.train.weak.10.dat -test n.test.weak.10.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/n.train.10.log$var -test_logfile NOlog/n.test.10.log$var -norm NOlog/n.norm.10 -VERBOSE 1
./LambdaMF -D 100 -train n.train.weak.20.dat -test n.test.weak.20.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/n.train.20.log$var -test_logfile NOlog/n.test.20.log$var -norm NOlog/n.norm.20 -VERBOSE 1
./LambdaMF -D 100 -train n.train.weak.50.dat -test n.test.weak.50.dat -a 0.0 -e 0.001 -n 250 -train_logfile NOlog/n.train.50.log$var -test_logfile NOlog/n.test.50.log$var -norm NOlog/n.norm.50 -VERBOSE 1
