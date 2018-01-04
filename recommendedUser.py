from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import pickle
from collections import defaultdict

import json

path_data='/Users/prateeksharma/Desktop/Machine Learning Project/dataset/'
yelp_files=['business','checkin','photos','review','tip','user']
review_file=yelp_files[3]
user_file=yelp_files[5]
# we have, 50000 reviews for training, we will have the next 5000 of them for testing to see weather people go to the
# restaurant  we have predicted and test the accuracy of our prediction
file_test=path_data+review_file+'.json'
file_user=path_data+user_file+'.json'

users_review=defaultdict(list)

with open(file_test) as yelp_review:
    ## read from 5001 line as first 5000 ha been used for testing and training
    f1 = yelp_review.read().strip().split("\n")[5001:]

review_act_user=''
for id, line in enumerate(f1):
    ## reading the 5001 line to recommend
    if (id+1)==5001:
        temp=json.loads(line)
        x= temp['text'].encode('utf-8').replace(".","").replace("/"," ").replace("!","").replace("\n","").replace(',','').split(' ')
        ## the business this user is going to
        actual_user = temp['user_id'].encode('utf-8')
        review_act_user=temp['text'].encode('utf-8')



## the text for which recommendation has to be made
texts=[]
temp = ' '.join(word for word in x)
texts.append(temp)


# load the tokenizer and the model
with open("keras_tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# load the encoded users which can be used to turn back the integer of user to actual user
with open("keras_users_encode.pickle", "rb") as f:
    encoded_users = pickle.load(f)

#load the encode to ranfor back the integer to user
with open("keras_encoder.pickle", "rb") as f:
    encoder= pickle.load(f)

model = load_model("yelp_recomender_model.hdf5")

## for the user you wanna get

# note that we shouldn't call "fit" on the tokenizer again
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=300)

# get predictions for each of your new texts
predictions = model.predict(data)
# print(predictions)
# print(texts)
# print(newtexts)
## converting class labels back to integers and which will then converted back to the integer users which they were
user_integer = np.argmax(predictions, axis=1)
#print(user_integer)
## getting the user for this class label i.e. the predicted user
user=encoder.inverse_transform(user_integer)

with open(file_user) as yelp_user:
    ## read the file
    f2 = yelp_user.read().strip().split("\n")

## Now getting the details of the predicted user
for id, line in enumerate(f2):
    temp=json.loads(line)
    uid=str(temp['user_id'].encode('utf-8'))
    if(user==uid):
        p_user_name=temp['name'].encode('utf-8')
        p_user_review_count = temp['review_count']
        p_user_yelping_since=temp['yelping_since'].encode('utf-8')
        p_user_useful=temp['useful']
        p_user_friends=temp['friends']
    elif(actual_user==uid):
        user_name = temp['name'].encode('utf-8')
        user_review_count = temp['review_count']
        user_yelping_since = temp['yelping_since'].encode('utf-8')
        user_useful = temp['useful']
        user_friends=temp['friends']





## Printing the actual business and the predicted business for the given text
print("Actual User who gave the review \n")
print "The name of the user is ", user_name,", he/she has given ",user_review_count," number of reviews."," He has been yelping since",  user_yelping_since, " and has given", user_useful," useful reviews."," Also he/she has following friends: ",user_friends

print("\n Similar user for this user \n")
print "The name of the user is ", p_user_name,", he/she has given ",p_user_review_count," number of reviews."," He has been yelping since",  p_user_yelping_since, " and has given", p_user_useful," useful reviews."," Also he/she has following friends: ",p_user_friends


review_pred_user=''
for id, line in enumerate(f1):
    temp=json.loads(line)
    uid=str(temp['user_id'].encode('utf-8'))
    if(user==uid):
        review_pred_user=temp['text'].encode('utf-8')

print "\n Review of Actual User: \n"
print review_act_user
print "\n Review of Predicted User: \n"
print review_pred_user


