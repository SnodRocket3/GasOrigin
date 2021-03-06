# import packages
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# change directory to data folder
os.chdir(os.getcwd() + r'/data/')

# code to pickle excel file for dataset update
# execute the code below that begins with ##
# remove every classified origin that doesn't contain: abiotic, thermogenic,
# primary microbial CO2 reduction, primary microbial acetate fermentation and secondary microbial
# load excel file and create a df
## df = pd.read_excel('Updated_Excel_File.xlsx')
# pickle the df
# df = pd.to_pickle(df, 'Natural_Gas_df_NewVersion(#.0).pkl')
# building Gas Database dataframe from a pickled df to increase speed
df = pd.read_pickle('Natural_Gas_df_NewVersion(#.0).pkl')

# the feature input is currently being worked on
# creating a list of possible input features to chose from
# list needs to be condensed, condensed list will be ongoing as research continues
for col in df.columns:
    print(col)
# requesting the desired features for natural gas origin classification
# the split function is so that the computer will recognize each entry as an individual string
command = input('''
From the above list, what geochemical data would you like
to use to analyze the origin of your natural gas sample?
Example of input(copy and paste from list, add spaces 
in-between): δ13CO2 δ13C1 C1/(C2+C3) δDC1
Keep in mind reduced dimensions typically results in larger
training datasets. You will be provided with training dataset
size and feature importance to help decision making!''').split()

# Origin is a column in the df of just the classified gas samples (training dataset with holes)
# adding the Origin column from our df to the beginning of the the input geochemistry feature list
# need to add the Origin column to determine holes in the dataset and create the training dataset df
command.insert(0, 'Origin')
# removing every row that contains a NaN
training_dataset_df = df[command].dropna()
# 300 was picked based on the interpretation of multiple CV curves
if len(training_dataset_df) <= 300:
    print('''Training dataset size is less than or equal to 300 try again and remove 
    some geochemical features or use some of the suggested features''')
# .pop removes the first item in the command list that was added above ('Origin')
command.pop(0)
# just the decided geochemical features in training dataset df (i.e. no NaNs)
X = training_dataset_df[command]
# all classified gas samples in the training dataset df
y = training_dataset_df[['Origin']]

# splits matrices into random test train subjects with the test size being 1/3, 2/3 = train size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# want the user to understand the size of their training group with the decided input features
# the initial training group is split to determine model accuracy but the train test split df is put back together
# when classifying the samples
print('Length of Training Samples: ', len(X))
# the 0.20 RandomForest Classifier changes the default n_estimators from 10 -> 100
# reassigning the RandomForestClassifier model with 50 n_estimators
clf = RandomForestClassifier(n_estimators=50)
# training the model using our created training dataset
# .values will give the values in an array (shape: (n,1)
# .ravel will convert that array shape to (n, )
# fitting the training dataset to the clf model
clf_m = clf.fit(X_train, y_train.values.ravel())
# running split test date in the RandomForestClassifier model
y_predict = clf_m.predict(X_test)
# display model accuracy
# normalize=True
# sample_weight=None
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))

# confusion matrix: in depth analyze of model accuracy
conf_mat = confusion_matrix(y_test, y_predict)
print(conf_mat)

# numerical representation of feature importance
# visual representation doesn't work/two attempts below, do not run them, pick numerical or visual
for feature in zip(X, clf.feature_importances_):
    print('Feature Importance: ', feature)

# by now the user should be able to input any desired features to create however many
# dimensions they want for the analysis. With the unique multi dimension entry the user will
# receive the size of the training data set as well as a list of feature importance. From
# there it is up to the user to determine if they would like to drop a non important feature
# so that the training dataset could increase, therefore potentially increasing the accuracy of
# the model even though the user would be reducing dimensions

# place the entire train and test group back into the model for a larger training group when classifying data entries
# override clf_m with the entire training dataset, not just the train set above
clf_m = clf.fit(X, y.values.ravel())

# code to input the geochemical values from lab you are trying to classify
classifying_input_data = input(
    "Input your geochemical values in order to determine the origin of your gas" + str(command)).split()
# creating a df of the lab data so that it can be inputted into the created model
lab_df = pd.DataFrame(classifying_input_data).T
# running lab data through the classifier model
results = clf_m.predict(lab_df)
# printing results of lab geochemistry ran through classifier model
print('Your natural gas sample is ', results, 'in origin')

# calculation the confidence for each classification
confidence = clf_m.predict_proba(lab_df)
# re-arranging the array to print classification values
abiotic_c = confidence[0,0]
microb_p_c_c = confidence[0,1]
microb_p_af_c = confidence[0,2]
s_microb = confidence[0,3]
thermogenic = confidence[0,4]
# print the confidence for each classification. It is the calculation of the number of trees voted on each
# classification over n_trees
print ('Abiotic:', abiotic_c*100,'%')
print("Microbial Primary (CO2 Reduction):", microb_p_c_c*100,'%')
print("Microbial Primary (methyl-type fermentation):", microb_p_af_c*100,'%')
print("Secondary Microbial:", s_microb*100,'%')
print("Thermogenic:", thermogenic*100,'%')
