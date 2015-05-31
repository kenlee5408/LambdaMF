LambdaMF
A novel collaborative filtering method incorporating lambda.

This implementation optimizes NDCG

Data format should be:(seperated by space)

user1 item1 rating1

user2 item2 rating2

... ... ...

//////////////////////////////////////////////////

compile and produce dataset:

download MovieLens100K data here:

	http://grouplens.org/datasets/movielens/
	
download Netflix data here (we are not responsible for this link):
	
	http://www.lifecrunch.biz/wp-content/uploads/2011/04/nf_prize_dataset.tar.gz

place the MovieLens100K dataset(u.data) and Netflix dataset(training_set/) in the same directory and run:

	sh prepare.sh
	mkdir MSElog
	mkdir L2log
	mkdir NOlog

run experiments of MSE regularization in paper:

	sh run_MSE_experiment.sh	

run experiments of L2 regularization in paper:

	sh run_L2_experiment.sh	

run experiments of NO regularization in paper(only norm reported):

	sh run_no_regularization_experiment.sh	

log file of training/testing/model norm will be in MSElog/ , L2log/ , NOlog/

you can use tail (in linux environment) command to see the result, for example:

	tail MSElog/m.test.50.log10

general usage:

	./LambdaMF -train [training data] -test [testing data] -e [eta] -L2 [L2 regularization coefficient(optional, default=0)] -n [number of iterations] -a [alpha] -train_logfile [log filename(optional)] -test_logfile [log filename(optional)] -norm [max of model norm log file(optional)]

If you want to test LambdaMF on other datasets, please set the right USER_N(number of users) and ITEM_N(number of items) in LambdaMF.h and recompile it.

The parameters used in paper:

MSE regularization: ./LambdaMF -train ... -test ... -a 0.5 -e 0.001 -n 250

L2 regularization: ./LambdaMF -train ... -test ... -a 0. -L2 0.5 -e 0.001 -n 250

