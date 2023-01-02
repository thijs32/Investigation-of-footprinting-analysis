import pyBigWig
import pandas as pd
from pybedtools import BedTool
import numpy as np
import math


#Get positive dataset ready for training the CNN

unibind_footprints = pd.read_csv('positive_set_footprints.bed', sep='\t',header=None)
unibind_footprints_test = unibind_footprints[unibind_footprints[0].isin(['chr8','chr9','chr13','chr14'])]
unibind_footprints_train = unibind_footprints[~unibind_footprints[0].isin(['chr8','chr9','chr13','chr14'])]
unibind_footprints_train = unibind_footprints_train[3] #select only the footprint scores from the dataframe
unibind_footprints_train = unibind_footprints_train.reset_index(drop=True)
#transform the literal values into arrays
for i in range(0,len(unibind_footprints_train)):
    unibind_footprints_train[i] = ast.literal_eval(unibind_footprints_train[i])
    unibind_footprints_train[i] = np.array(unibind_footprints_train[i])

#Get negative dataset ready for training the CNN

negative_footprints = pd.read_csv('negative_set_footprints.bed', sep='\t',header=None)
negative_footprints_test = negative_footprints[negative_footprints[0].isin(['chr8','chr9','chr13','chr14'])]
negative_test_copy = negative_footprints_test
negative_footprints_train = negative_footprints[~negative_footprints[0].isin(['chr8','chr9','chr13','chr14'])]
negative_footprints_train = negative_footprints_train[3]
negative_footprints_train = negative_footprints_train.reset_index(drop=True)
for i in range(0,len(negative_footprints_train)):
    negative_footprints_train[i] = ast.literal_eval(negative_footprints_train[i])
    negative_footprints_train[i] = np.array(negative_footprints_train[i])

#concatenate the negative and positive data and turn into tensors
train_X = np.concatenate((unibind_footprints_train,negative_footprints_train),axis=0)
train_X = pd.Series(train_X)
train_X= train_X.to_list()
train_X=np.asarray(train_X).astype(np.float32)
train_X = tf.constant(train_X)
train_X = tf.expand_dims(train_X,axis=-1)

#make y-labels for the training data and turn into tensors
ones_train = np.ones(len(unibind_footprints_train))
zeros_train = np.zeros(len(negative_footprints_train))
ones_train_comp = np.zeros(len(unibind_footprints_train))
zeros_train_comp = np.ones(len(negative_footprints_train))
train_y_step1 = np.column_stack((ones_train,ones_train_comp))
train_y_step2 = np.column_stack((zeros_train,zeros_train_comp))
train_y = np.concatenate((train_y_step1,train_y_step2),axis = 0 )
train_y = tf.constant(train_y)
train_y = tf.expand_dims(train_y,axis=-1)

#Example of CNN used with windowsize 500

model = keras.Sequential(
    [
        layers.Conv1D(filters=64, kernel_size=12,input_shape=(501,1) , activation="relu"),
        layers.Conv1D(filters=64, kernel_size=12, activation="relu"),
        layers.MaxPooling1D(pool_size=3),
        layers.Dropout(0.25),
        layers.Conv1D(filters=32, kernel_size=2, activation="relu"),
        #layers.Conv1D(filters=32, kernel_size=12, activation="relu"),
        layers.MaxPooling1D(pool_size=3),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(15, activation="relu"),
        layers.Dense(2, activation="softmax"),

    ]
)

model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="binary_accuracy")
#train and save the model
history = model.fit(train_X, train_y, epochs=10, batch_size=32,validation_split=0.2)
model.save("CNN_footprint_500bp.hdf5")

#To test the model on a test data set and extract AUC

model_prediction500 = model500.predict(test_X500)
fpr, tpr, thresholds = metrics.roc_curve(test_y[:,0], model_prediction500[:,0], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

#logistic regression example

LR_foot = LogisticRegression()
scoreshape_foot = np.ravel(train_X)
scoreshape_foot = scoreshape_foot.reshape(-1,1)
LR_foot.fit(scoreshape_foot ,train_y[:,0])
#make predictions and extract the coefficients
predictfoot = LR_foot.predict(scoreshape_foot)
predict_probs_foot = LR_foot.predict_proba(scoreshape_foot)[:,1]
LR_foot.coef_
LR_foot.intercept_
#Threshold:
LR_foot.intercept_ / (0 - LR_foot.coef_)

#to test logistic regression on test set and extract AUC
fprfoot, tprfoot, thresholdsfoot = metrics.roc_curve(test_y[:,0], predict_probs_foot, pos_label=1)
roc_aucfoot = metrics.auc(fprfoot, tprfoot)