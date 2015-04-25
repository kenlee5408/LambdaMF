from random import shuffle
from collections import defaultdict

arr = []

itemcount = 1
itemmap = defaultdict()
usercount = 0
usermap = defaultdict()
train_num = [10,20,50]
total_thre = [20,30,60]

train_ratings = defaultdict()
test_ratings = defaultdict()

filenames = []
with open('filenames') as fp:
	for line in fp:
		filenames.append(line.strip())
		
prefix = 'training_set/'
for item,file in enumerate(filenames):
	fp = open(prefix+file)
	fp.readline()
	item = str(item)

	for line in fp:
		line = line.strip().split(',')
		if line[0] not in usermap:
			usermap[line[0]] = str(usercount)
			usercount+=1
		line[0] = usermap[line[0]]
		if line[0] not in train_ratings:
			train_ratings[line[0]] = []
		train_ratings[line[0]].append(item+':'+line[1])

for i in xrange(3):	
	with open('n.train.weak.'+str(train_num[i])+'.dat','w') as trainfp,open('n.test.weak.'+str(train_num[i])+'.dat','w') as testfp:
		print 'preparing','n.train.weak.'+str(train_num[i])+'.dat','n.test.weak.'+str(train_num[i])+'.dat'
		for u in train_ratings:
			if len(train_ratings[u]) < total_thre[i]:
				continue
				
			shuffle(train_ratings[u])
			for k in train_ratings[u][:train_num[i]]:
				item = k.split(':')
				trainfp.write(u+' '+item[0]+' '+item[1]+'\n')
			for k in train_ratings[u][train_num[i]:]:
				item = k.split(':')
				testfp.write(u+' '+item[0]+' '+item[1]+'\n')
print usercount,'users' #m: 480189 users
print len(filenames),'items' #m: 17770 items