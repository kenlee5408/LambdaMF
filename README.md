LambdaMF
A novel collaborative filtering method incorporating lambda.

This implementation optimizes NDCG on linux environment.

The codes are written using C++, python and shell commands.

Data format should be:(seperated by space)

user_id1 item_id1 rating1

user_id2 item_id2 rating2

... ... ...

//////////////////////////////////////////////////

compile the code:

	sh compile.sh

download MovieLens100K data here:

	http://grouplens.org/datasets/movielens/
	
download Netflix data here (we are not responsible for this link):
	
	http://www.lifecrunch.biz/wp-content/uploads/2011/04/nf_prize_dataset.tar.gz

place the MovieLens100K dataset(u.data) and Netflix dataset(training_set/) in the same directory and run:

	sh prepare.sh
	mkdir MSElog
	mkdir L2log
	mkdir NOlog
	mkdir log

run experiments of MSE regularization in paper(it takes about half day):

	sh run_MSE_experiment.sh	

run experiments of L2 regularization in paper(it takes about half day):

	sh run_L2_experiment.sh	

run experiments of NO regularization in paper(only norm reported in paper)(it takes about half day):

	sh run_no_regularization_experiment.sh	

run experiments of checking the optimization and optimum of LambdaMF with MSE regularization:

	sh test_experiment.sh 

This version (test_experiment.sh) stops only if NDCG@10 == 1.0 using C++ internal double precision, so it is more reliable than using the log file. 


All log file of training, testing, model_norm and convergence will be in MSElog/ , L2log/ , NOlog/ and log/

you can use tail or vim (in linux environment) command to see the result, for example:

	tail MSElog/m.test.50.log10

general usage:

	./LambdaMF [Parameter flag] [Parameter value]

Legitimate parameter flags/ parameter value:

	-train/ training data filename(string)(necessary)
	-test	/ testing data filename(string)(necessary)
	-e		/ eta, learning rate(double)(default=0.001)
	-L2		/ L2 regularization coefficient(double)(default=0.)
	-a		/ alpha, MSE regularization coefficient(double)(default=0.5)
	-n		/ number of iterations(int)(default=250)
	-D		/	dimension of latent factors(int)(default=100)
	-predict	/	predict filename on testing data(string)(optional)
	-VERBOSE	/	verbose levels(-1/0/1/2)(optional)
	-train_logfile	/	log filename for training data(string)(optional)
	-test_logfile		/	log filename for testing data(string)(optional)

The parameters used in paper:

MSE regularization: ./LambdaMF -train [...] -test [...] -a 0.5 -e 0.001 -n 250

L2 regularization: ./LambdaMF -train [...] -test [...] -a 0. -L2 0.5 -e 0.001 -n 250

