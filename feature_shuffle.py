#! /usr/bin/env python
# coding = utf-8

import json
import sys
import numpy as np
from sklearn.decomposition import PCA

def filter():
	feature_set = dict()
	feature_dic = dict()

	with open('./train.json') as f:
		json_str = f.read().strip()
		try:
			data = json.loads(json_str)
		except:
			sys.stderr.write('Error when load json')
			exit(1)
		feature_map = data['features']
		for idx, lists in feature_map.iteritems():
			if idx not in feature_dic:
				feature_dic[idx] = list()
			for feature in lists:
				feature = feature.strip()
				fea_words = feature.split()
				if len(fea_words) > 3:
					continue
				feature = ''.join(fea_words).lower()
				feature_dic[idx].append(feature)
				if feature in feature_set:
					feature_set[feature] += 1
				else:
					feature_set[feature] = 1

	#feature_num = len(feature_set)
	#user_num = len(feature_dic)

	#print 'Total number of feature is %s , number of user is %s' %(str(feature_num), str(user_num))
	#f = open('./name', 'w')

	top_fea = sorted(feature_set.iteritems(), key=lambda x:x[1], reverse=True)[:50]

	#thres = user_num / 200
	user_vec = {}
	for tup in top_fea:

		feature = tup[0]
		for userid, fealist in feature_dic.iteritems():
			if userid not in user_vec:
				user_vec[userid] = []
			if feature in fealist:
				user_vec[userid].append(1)
			else:
				user_vec[userid].append(0)

	user_mat = []
	user_id = []
	for uid, vec in user_vec.iteritems():
		user_id.append(int(uid))
		user_mat.append(vec)
	user_mat = np.array(user_mat)

	pca = PCA(n_components=30)
	fea_mat = pca.fit_transform(user_mat)

	return user_id, fea_mat


if __name__ == '__main__':
	filter()


			
