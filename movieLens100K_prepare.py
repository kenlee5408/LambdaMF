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

with open('u.data') as fp:
	for line in fp:
		line = line.strip().split('\t')
		if line[1] not in itemmap:
			itemmap[line[1]] = str(itemcount)
			itemcount+=1
		if line[0] not in usermap:
			usermap[line[0]] = str(usercount)
			usercount+=1
		line[0] = usermap[line[0]]
		line[1] = itemmap[line[1]]
		if line[0] not in train_ratings:
			train_ratings[line[0]] = []
		train_ratings[line[0]].append(line[1]+':'+line[2])

for i in xrange(3):	
	with open('m.train.weak.'+str(train_num[i])+'.dat','w') as trainfp,open('m.test.weak.'+str(train_num[i])+'.dat','w') as testfp:
		print 'preparing','m.train.weak.'+str(train_num[i])+'.dat','m.test.weak.'+str(train_num[i])+'.dat'
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
print usercount,'users' #943 users
print itemcount-1,'items' #1682 items