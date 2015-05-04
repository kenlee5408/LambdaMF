LambdaMF
A novel collaborative filtering method incorporating lambda.

This implementation optimizes NDCG

Data format should be:(seperated by space)

user1 item1 rating1

user2 item2 rating2

... ... ...

//////////////////////////////////////////////////

compile and produce dataset:

place the MovieLens100K dataset(u.data) and Netflix dataset(training_set/) in the same directory and run:

	sh prepare.sh
	mkdir log

run experiments of MSE regularization in paper:

	sh run_experiment.sh	
	
log file of testing dataset will be in log/

general usage:

	./LambdaMF -train [training data] -test [testing data] -e [eta] -L2 [L2 regularization coefficient(optional, default=0)] -n [number of iterations] -a [alpha] -train_logfile [log filename(optional)] -test_logfile [log filename(optional)]

The parameters used in paper:

MSE regularization: ./LambdaMF -train . -test . -a 0.5 -e 0.001 -n 250

L2 regularization: ./LambdaMF -train . -test . -a 0. -L2 0.5 -e 0.001 -n 250

