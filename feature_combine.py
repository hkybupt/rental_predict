#!/usr/bin/env python
# coding = 'utf-8'

import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import sys
from feature_shuffle import filter
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from matplotlib.pyplot import plot

def shuffle():
	df = pd.read_json(open('./train.json', 'r'))
	df['num_photos'] = df['photos'].apply(len)
	df['num_features'] = df['features'].apply(len)
	df['num_description_words'] = df['description'].apply(lambda x: len(x.split()))
	df['created'] = pd.to_datetime(df['created'])
	df['created_year'] = df['created'].dt.year
	df['created_month'] = df['created'].dt.month
	df['created_day'] = df['created'].dt.day
	uid, umat = filter()
	fea_col = ['fea_'+str(i) for i in range(30)]
	fea_df = pd.DataFrame(umat, index=uid, columns=fea_col)
	df = pd.concat([df, fea_df], axis=1)

	features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price","num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day"]
	features_to_use.extend(fea_col)
	lbl = preprocessing.LabelEncoder()
	lbl.fit(list(df['manager_id'].values))
	df['manager_id'] = lbl.transform(list(df['manager_id'].values))
	features_to_use.append('manager_id')

	X = df[features_to_use]
	y = df['interest_level']

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

	temp = pd.concat([X_train.manager_id, pd.get_dummies(y_train)], axis=1).groupby('manager_id').mean()
	temp.columns = ['high_frac', 'low_frac', 'medium_frac']
	temp['count'] = X_train.groupby('manager_id').count().iloc[:,1]
	temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']
	unranked_managers_ixes = temp['count']<20
	ranked_managers_ixes = ~unranked_managers_ixes
	mean_values = temp.loc[ranked_managers_ixes, ['high_frac','low_frac', 'medium_frac','manager_skill']].mean()
	temp.loc[unranked_managers_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
	X_train = X_train.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
	X_val = X_val.merge(temp.reset_index(),how='left', left_on='manager_id', right_on='manager_id')
	new_manager_ixes = X_val['high_frac'].isnull()
	X_val.loc[new_manager_ixes,['high_frac','low_frac', 'medium_frac','manager_skill']] = mean_values.values
	features_to_use.append('manager_skill')

	n_param = range(800, 1500, 50)
	for n_estimator in n_param:
		clf = RandomForestClassifier(n_estimators=n_estimator)
		clf.fit(X_train[features_to_use], y_train)
		y_val_pred = clf.predict_proba(X_val[features_to_use])
		y_train_pred = clf.predict_proba(X_train[features_to_use])
		#train_acc = (y_train == clf.predict(X_train[features_to_use])).mean()
		#val_acc = (y_val == clf.predict(X_val[features_to_use])).mean()
		loss = log_loss(y_val, y_val_pred)
		print 'using estimator', str(n_estimator),'test loss ', str(loss)
		
	pd.Series(index = features_to_use, data = clf.feature_importances_).sort_values().plot(kind = 'bar')


if __name__ == '__main__':
	shuffle()











