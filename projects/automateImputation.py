from sklearn.preprocessing import OrdinalEncoder
from fancyimpute import KNN
import numpy as np
def automate_imputation(users):

	# Create an empty dictionary ordinal_enc_dict
	ordinal_enc_dict = {}

	for col_name in users:
		# Create Ordinal encoder for col
	    # print(col_name)
	    ordinal_enc_dict[col_name] = OrdinalEncoder()
	    col = users[col_name]
	    
	    # Select non-null values of col
	    col_not_null = col[col.notnull()]
	    reshaped_vals = col_not_null.values.reshape(-1, 1)
	    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
	    
	    # Select the non-null values for the column col_name in users and store the encoded values
	    users.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
	# print(users)
	# users = users.astype(float)
	users_KNN_imputed = users.copy(deep = True)
	# Create Mice imputer
	KNN_imputer = KNN()

	# Impute 'users' DataFrame. It is rounded to get integer values
	users_KNN_imputed.iloc[:, :] = np.round(KNN_imputer.fit_transform(users))

	# Loop over the column names in 'users'
	for col_name in users_KNN_imputed:
	    # print(col_name)
	    # Reshape the column data
	    reshaped = users_KNN_imputed[col_name].values.reshape(-1, 1)
	    
	    # Select the column's Encoder and perform inverse transform on 'reshaped'
	    users_KNN_imputed[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped).ravel()
	# print(users_KNN_imputed)
	return users_KNN_imputed

if __name__=="__main__":
	import pandas as pd
	path = r"C:\Users\gprak\Downloads\projects\Data\titanic.csv"
	df = pd.read_csv(path)
	rv_df = automate_imputation(df)
	print(rv_df)