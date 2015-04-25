#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<map>
#include<set>
#include<ctime>
#include<string>
#include<vector>
#include<utility>
#include<algorithm>
#include<omp.h>
#include<iterator>
#include<queue>

#define USER_N 480190
#define ITEM_N 17771
#define D 100

char train_filename[100];
char test_filename[100];
char *train_logfile = NULL;
char *test_logfile = NULL;

double ETA;
double LAMBDA = 0;
double ALPHA = 0.1;
int n = 1;

// hard coded
double U[480190][D];
double V[17771][D];

using namespace std;

map<int,int> usermap;
set<int> userSeen[USER_N];
set<int> itemSeen;
int userCount;
vector<pair<int,double> > data[USER_N];
vector<pair<int,double> > testdata[USER_N];
int totalCount;

inline double sqr(double x){
	return x*x;
}
inline double g(double x){
	return 1/(1+exp(-x));
}
inline double dg(double x){
	return exp(x)/sqr(1+exp(x));
}
bool cmp(const pair<int,double> &a, const pair<int,double> &b){
	return a.second > b.second;
}

bool cmp2(const pair<int,double> &a, const pair<int,double> &b){
	return a.second < b.second;
}

class Item{
public:
	double ranking;
	int ind;
	double pred;
	double rating;
	Item(double a, int b, double c, double d){
		ranking = a;	ind = b;	pred = c;	rating = d;
	}
};
class ComparePair{
public:
	bool operator()(pair<int,double> &a, pair<int,double> &b){
		return a.second < b.second;
	}
};

bool cmpPred(const Item &a, const Item &b){
	return a.pred > b.pred;
}
bool cmpRating(const Item &a, const Item &b){
	return a.rating > b.rating;
}

void readInput(int argc,char* argv[]){
	for(int i=1;i<argc;i++){
		if(!strcmp(argv[i],"-train")){
			strcpy(train_filename,argv[i+1]);
		}else if(!strcmp(argv[i],"-test")){
			strcpy(test_filename,argv[i+1]);
		}else if(!strcmp(argv[i],"-e")){
			ETA = atof(argv[i+1]);
		}else if(!strcmp(argv[i],"-L2")){
			LAMBDA = atof(argv[i+1]);
		}else if(!strcmp(argv[i],"-n")){
			n = atoi(argv[i+1]);
		}else if(!strcmp(argv[i],"-a")){
			ALPHA = atof(argv[i+1]);
		}else if(!strcmp(argv[i],"-train_logfile")){
			train_logfile = argv[i+1];
		}else if(!strcmp(argv[i],"-test_logfile")){
			test_logfile = argv[i+1];
		}
	}
	printf("train file: %s\ntest file: %s\nETA: %lf\nALPHA: %lf\niteration: %d\n",train_filename,test_filename,ETA,ALPHA,n);
}
void initialize(){
	srand(time(NULL));
	for(int i=0;i<USER_N;i++)
		for(int j=0;j<D;j++)
			U[i][j] = ((double) rand() / (RAND_MAX)) * 0.1;
	for(int i=0;i<ITEM_N;i++)
		for(int j=0;j<D;j++)
			V[i][j] = ((double) rand() / (RAND_MAX)) * 0.1;
}

void readData(char filename[80],bool isTest){
	FILE* fp = fopen(filename,"r");
	int user,item,rating;
	map<int,int>::iterator it;
	while(fscanf(fp,"%d%d%d",&user,&item,&rating)!=EOF){
		it = usermap.find(user);
		if(it == usermap.end()){
			usermap[user] = userCount;
			user = userCount++;
		}
		else
			user = it->second;
		if(isTest)
			testdata[user].push_back(pair<int,double>(item,double(rating)));
		else
			data[user].push_back(pair<int,double>(item,double(rating)));
	}
	return;
}
double dot(double a[D], double b[D]){
	double temp = 0;
	for(int i=0;i<D;i++)
		temp += a[i]*b[i];
	return temp;
}

inline double norm(double M[D]){
	return sqrt(dot(M,M));
}
bool found(int item,vector<int> arr){
	for(int i=0;i<arr.size();i++)
		if(arr[i]==item)
			return true;
	return false;
}

void updateModel(int iterN){
	int *items = new int[itemSeen.size()];
	priority_queue<pair<int,double>,vector<pair<int,double> >,ComparePair> score;
	
	
	set<int>::iterator it = itemSeen.begin();
	for(int k=0;k<itemSeen.size();k++){
		items[k] = *it;
		advance(it,1);
	}
	
	for(int h=0;h<userCount;h++){
		//int u = rand() % userCount;
		int u = h;
		vector<Item> candidate;
		for(int i=0;i<data[u].size();i++)
			candidate.push_back(Item(-1,data[u][i].first,dot(U[u],V[data[u][i].first]),data[u][i].second));
		
		sort(candidate.begin(),candidate.end(),cmpRating);
		double N = 0;
		for(int i=0;i<candidate.size();i++)
			N+=(pow(2,candidate[i].rating) - 1)/log2(2+i);
		N = 1/N;
		// calculate maxNDCG
		sort(candidate.begin(),candidate.end(),cmpPred);
		
		double dU[D];
		for(int k=0;k<D;k++)
			dU[k] = -LAMBDA * U[u][k];	// if taking L2 regularizor
			//dU[k] = 0;
		//double** dV = new double*[candidate.size()];
		
		double** dV = new double*[candidate.size()];
		for(int i=0;i<candidate.size();i++){
			dV[i] = new double[D];
			for(int k=0;k<D;k++)
				dV[i][k] = -LAMBDA * V[candidate[i].ind][k];	// if taking L2 regularizor
				//dV[i][k] = 0;
		}
				
		for(int a=0;a<candidate.size();a++){
			for(int b=a+1;b<candidate.size();b++){
				if(a==b)
					continue;
				int i,j;
				if(candidate[a].rating > candidate[b].rating){
					i = a;	j = b;
				}else if(candidate[a].rating < candidate[b].rating){
					i = b;	j = a;
				}
				else
					continue;
					
				double dCdS = abs(N *(pow(2.,candidate[i].rating)-pow(2.,candidate[j].rating))*(1./log2(2+i)-1./log2(2+j)));
				//double dCdS = 0.;
				//if(i < j)
				//	continue;
				//double dCdS = N *(pow(2.,candidate[i].rating)-pow(2.,candidate[j].rating))*(1./log2(2+j)-1./log2(2+i));
				double temp[D];
				for(int k=0;k<D;k++)
					temp[k] = V[i][k] - V[j][k];
					
				//double z = exp(-dot(U[u],temp))/(1+exp(-dot(U[u],temp)));
				//double z = (1./(1+exp(candidate[i].pred-candidate[j].pred)));
				double z = 1;
				
				for(int k=0;k<D;k++){
					dU[k] += dCdS * z * (V[candidate[i].ind][k] - V[candidate[j].ind][k]);// - LAMBDA * U[u][k];
					dV[i][k] += dCdS * z * U[u][k];// - LAMBDA * V[i][k];
					dV[j][k] += - dCdS * z * U[u][k];// - LAMBDA * V[j][k];
				}
			}
		}
		
		for(int a=0;a<candidate.size();a++){
			for(int k=0;k<D;k++){
				dU[k] -= ALPHA * (candidate[a].pred - candidate[a].rating) * V[candidate[a].ind][k];
				dV[a][k] -= ALPHA * (candidate[a].pred - candidate[a].rating) * U[u][k];
			}
		}
		
		for(int a=0;a<candidate.size();a++)
			for(int k=0;k<D;k++){
				V[candidate[a].ind][k] += ETA*dV[a][k];
			}
		for(int k=0;k<D;k++){
			U[u][k] += ETA*dU[k];
		}
		
		for(int i=0;i<candidate.size();i++)
			delete[] dV[i];
		delete[] dV;
		
	}

	// used for detecting overflow
	/*if(iterN!=0 && iterN%50==0){
		double normMax = 0;
		for(int i=0;i<userCount;i++){
			double temp = norm(U[i]);
			normMax = (temp > normMax) ? temp : normMax;
		}
		for(int i=0;i<ITEM_N;i++){
			double temp = norm(V[i]);
			normMax = (temp > normMax) ? temp : normMax;
		}
		printf("norm max = %lf\n",normMax);
	}*/
}

void evaluate(vector<pair<int,double> > usedata[USER_N],int u,double &ndcg){
	vector<Item> candidate;
	for(int i=0;i<usedata[u].size();i++)
		candidate.push_back(Item(-1,usedata[u][i].first,dot(U[u],V[usedata[u][i].first]),usedata[u][i].second));
	
	sort(candidate.begin(),candidate.end(),cmpRating);
	double N = 0;
	double dcg = 0;
	for(int i=0;i<10;i++)
		N+=(pow(2,candidate[i].rating) - 1)/log2(2+i);
	sort(candidate.begin(),candidate.end(),cmpPred);
	for(int i=0;i<10;i++)
		dcg+=(pow(2,candidate[i].rating)-1)/log2(2+i);
	ndcg+=dcg/N;

}
void output_log(FILE* fp,vector<pair<int,double> > usedata[USER_N],int iterN){
	if(fp == NULL)
		return;
	double ndcg = 0;
	for(int i=0;i<userCount;i++)
		evaluate(usedata,i,ndcg);
	if(userCount!=0){
		ndcg/=userCount;
		fprintf(fp,"%d iteration, ndcg@10= %.4lf\n",iterN,ndcg);
	}
}
int main(int argc, char* argv[]){
	if(argc < 10){
		printf("usage: ./LambdaMF -train [train_data_name] -test [test_data_name] -n [N] -L2 [L2 coefficient] -a [ALPHA] -e [ETA]\n");
		return 0;
	}
	readInput(argc,argv);
	initialize();	// set random seed here, initialize U,V
	
	printf("reading training data:  %s\n",train_filename);
	readData(train_filename,false);	// encode user here
	for(int k=0;k<userCount;k++)
		for(int m=0;m<data[k].size();m++){
			itemSeen.insert(data[k][m].first);
			userSeen[k].insert(data[k][m].first);
		}
	//updateModel();	
	printf("%d users, %d items\n",userCount,itemSeen.size());
	printf("reading testing data:  %s\n",test_filename);
	readData(test_filename,true);	// encode user here
	
	FILE *trainlog = NULL;
	FILE *testlog = NULL;
	
	if(train_logfile!=NULL)
		trainlog = fopen(train_logfile,"w");
	if(test_logfile!=NULL)
		testlog = fopen(test_logfile,"w");
	
	vector<int> itemSet;
	for(set<int>::iterator it = itemSeen.begin();it!=itemSeen.end();++it)
		itemSet.push_back(*it);	// in order to use O(N) traverse later
	
	output_log(trainlog,data,0);
	output_log(testlog,testdata,0);
	
	time_t start,end;
	time(&start);
	for(int t=1;t<=n;t++){		
		updateModel(t);
	
		output_log(trainlog,data,t);
		output_log(testlog,testdata,t);
		double ndcgTrain = 0;
		double ndcgTest = 0;
		
		//if(t!=0 && t%50==0){
		if(t==n){
			time(&end);
			double seconds = difftime(end,start);
			printf("training LambdaMF takes %.f seconds\n",seconds);
			for(int i=0;i<userCount;i++){	// current seen user number equal to userCount, id is encoded
				evaluate(data,i,ndcgTrain);
				evaluate(testdata,i,ndcgTest);
			}
					
			if(userCount!=0){
				ndcgTrain/=userCount;
				ndcgTest/=userCount;
				printf("%d iteration, train ndcg@10= %.4lf, test ndcg@10= %.4lf\n",t,ndcgTrain,ndcgTest);
			}
		}
	}

	// if want to save the result matrix
	/*
	FILE *outfp = fopen("result.ndcg","w");
	for(int i=0;i<userCount;i++){
		for(int j=0;j<testdata[i].size();j++)
			fprintf(outfp,"%d:%f ",testdata[i][j].first,dot(U[i],V[testdata[i][j].first]));
		fprintf(outfp,"\n");
	}
	*/
	return 0;
}
