from datetime import datetime
import json
from collections import defaultdict


path_data='/Users/prateeksharma/Desktop/Machine Learning Project/dataset/'
yelp_files=['business','checkin','photos','review','tip','user']
review_file=yelp_files[3]
user_file=yelp_files[5]

file_review=path_data+review_file+'.json'
file_user=path_data+user_file+'.json'

t1 = datetime.now()

users_review=defaultdict(list)

with open(file_review) as yelp_review:
    f = yelp_review.read().strip().split("\n")


for id, line in enumerate(f):
    if (id+1)%40000==0: break

    temp=json.loads(line)
    x= temp['text'].encode('utf-8').replace(".","").replace("/"," ").replace("!","").replace("\n","").replace(',','').split(' ')
    user = temp['user_id'].encode('utf-8')

    users_review[user].extend(x)


t2=(datetime.now() - t1)
print '----data is read---',(t2)

texts = []
users_str = []

for user, review in users_review.iteritems():

    temp= ' '.join(word for word in review)
    texts.append(temp)
    users_str.append(user)

# print(users_review)
# print(texts[0])
# print(users_str[0])
import pickle
#
# pickle.dump(model, open('yelp_model.p','w'))

# save the tokenizer,users,encoded_users(in integers),encoder(so that it can be used to tranform back the integer to user) and model
with open("keras_text1.pickle", "wb") as f:
   pickle.dump(texts, f)

with open("keras_user1.pickle", "wb") as f:
   pickle.dump(users_str, f)