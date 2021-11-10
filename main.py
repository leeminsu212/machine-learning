import random
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

flu=['Y', 'N', 'Y', 'Y']
fever=['L', 'M', 'H', 'M']
sinus=['Y', 'N', 'Y', 'Y']
ache=['Y', 'N', 'N', 'N']
swell=['Y', 'N', 'Y', 'N']
headache=['N', 'N', 'Y', 'Y']

# function making random flu tuple
def makeFlu():
    for i in range(26):
        v=random.choice(['Y', 'N'])
        flu.append(v)

# function making random fever tuple
def makeFever():
    for i in range(26):
        v=random.choice(['L', 'M', 'H'])
        fever.append(v)

# function making temporary value of sinus, ache, swell and headache
# set least number of Y depends on fluYorN value
# if flu value is Y then fluYorN value is Y
def makeTempV(fluYorN):
    c=0
    while(c==0):
        count = 0
        tempV = []
        for i in range(4):
            v=random.choice(['Y', 'N'])
            if(v=='Y'):
                count+=1
            tempV.append(v)
        if(fluYorN=='Y'):
            if(count>=2):
                c=1
        if(fluYorN=='N'):
            if(count<=2):
                c=1
    # return list of sinus, ache, swell and headache
    return tempV

# function making random sinus, ache, swell and headache tuples
def makeSASH(flu):
    for i in range(26):
        tempV=makeTempV(flu[i+4])
        sinus.append(tempV[0])
        ache.append(tempV[1])
        swell.append(tempV[2])
        headache.append(tempV[3])

# make 26 more tuples
makeFlu()
makeFever()
makeSASH(flu)

df=pd.DataFrame({'flu':flu,
                 'fever':fever,
                 'sinus':sinus,
                 'ache':ache,
                 'swell':swell,
                 'headache':headache})
print(df)

# encoding all features
enc=LabelEncoder()
fluE=enc.fit_transform(flu)
feverE=enc.fit_transform(fever)
sinusE=enc.fit_transform(sinus)
acheE=enc.fit_transform(ache)
swellE=enc.fit_transform(swell)
headacheE=enc.fit_transform(headache)

# list for store zipped features
features=[]
# zip each features
for f in zip(feverE, sinusE, acheE, swellE, headacheE):
    features.append(f)

# build model
model=GaussianNB()
model.fit(features, fluE)
print('Score :', model.score(features, fluE))

# compare first 4 tuples actual flu value with predicted flu value
for i in range(4):
    print(i+1, 'actual flu :', flu[i])
    if(model.predict([features[i]])[0]==0):
        p='N'
    else:
        p='Y'
    print(i+1, 'predicted flu :', p, '\n')