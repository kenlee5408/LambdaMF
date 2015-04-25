import sys
import numpy as np

def collect(name):
	ndcg = []
	for i in xrange(1,11):
		with open(name+str(i)) as fp:
			for line in fp:
				line = line.strip().split(' ')
				score = line[-1]
			ndcg.append(np.float64(score))
	print '{0:.4f}'.format(np.mean(ndcg)),'{0:.4f}'.format(np.std(ndcg))

if __name__=='__main__':
	if len(sys.argv) != 2:
		print 'usage: python collectResult.py [log file prefix]'
                print 'e.g. python collectResult.py log/m.test.10.log'
	else:
		filename = sys.argv[1]
		collect(filename)
