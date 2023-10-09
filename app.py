import streamlit as st
import pandas as pd
import numpy as np
import os

import re
train= pd.read_csv("./train.csv", names=["place", "dnm", "suff"])
suffList = []
data = train
for i, row in data.iterrows():
    p: str = row["place"]
    d: str = row["dnm"]
    prefix = os.path.commonprefix([p, d])
    suffix = ""
    if(prefix.__len__() > 0):
        suffix = str.join("", d.split(prefix)[1:])

    # s = []
    # for i, c in enumerate(suffix):
    #     s.append(suffix[i:])
    data.at[i, "suff"] = suffix
    suffList.append(suffix)


def getTopK(suffList, hist):
    topSuff = []
    temp = [[hist[suffix] for suffix in suffixes] for suffixes in suffList]
    for i, suffixes in enumerate(suffList):
        copy = suffixes
        histVals = temp[i]
        topSuffixes = []
        for t in range(3):
            if(histVals.__len__() == 0):
                continue
            topIndex = np.argmax(histVals)
            topSuffix = copy[topIndex]
            topSuffixes.append(topSuffix)
            del copy[topIndex]
            del histVals[topIndex]
        topSuff.append(topSuffixes)
    return topSuff


def getHist2d(list2d):
    hist = {}
    for i, list1d in enumerate(list2d):
        for j, item in enumerate(list1d):
            if item in hist:
                hist[item] = hist[item] + 1
            else:
                hist[item] = 1
    return hist


def getHist1d(list1d):
    hist = {}
    for j, item in enumerate(list1d):
        if item in hist:
            hist[item] = hist[item] + 1
        else:
            hist[item] = 1
    return hist


def printDict(d):
    dict(sorted(d.items(), key=lambda item: item[1], reverse=True))


def getPL2Hist(data):
    pL2Dict = {}
    for i, row in data.iterrows():
        p: str = row["place"]
        d: str = row["dnm"]
        s: str = row["suff"]
        pL2: str = p[-2:]
        if not pL2 in pL2Dict:
            pL2Dict[pL2] = {}
        if s in pL2Dict[pL2]:
            pL2Dict[pL2][s] = pL2Dict[pL2][s] + 1
        else:
            pL2Dict[pL2][s] = 1
    return pL2Dict


def getPL2HistWithDiff(data):
    pL2Dict = {}
    for i, row in data.iterrows():
        p: str = row["place"]
        d: str = row["dnm"]
        s: str = row["suff"]
        pL2: str = p[-2:]
        if not pL2 in pL2Dict:
            pL2Dict[pL2] = {}
        if s in pL2Dict[pL2]:
            pL2Dict[pL2][s] = pL2Dict[pL2][s] + 1
        else:
            pL2Dict[pL2][s] = 1
    return pL2Dict


def orthographicFix(place: str, suffix: str):
    m = re.search("(.*)(os|ey)$", place)
    if(m):
        return m.group(1)

    m = re.search("(.*)(na|ng)$", place)
    if(m):
        return m.group(1) + "n"

    # m = re.search("(.*)n(e|s)$", place)
    # if(m):
    #     return m.group(1) + m.group(2)

    m = re.search("(.*)(l|u|nr|)(s|e)$", place)
    n = 0 if (pd.isna(suffix) or len(suffix) == 0)else re.search("^i", suffix)
    if(m and n):
        return m.group(1) + m.group(2)

    m = re.search("(.*)(o|y|s|w)$", place)
    if(m):
        return m.group(1)
    # m = re.search("(.*)(w)$", place)
    # if(m):
    #     return m.group(1) + "v"
    return place


def getMostSimilarRule(key, rulesDict):
    if key in rulesDict:
        return key
    for rule in rulesDict:
        if(rule[-1] == key[-1]):
            return rule
        
# def pL2DictToDF(pL2Dict: dict):
#     data = {}
#     for key in pL2Dict:
        


def predict(rules, data):
    acc = 0
    outlier = 0
    print("DATA SIZE", len(data))
    for i, row in data.iterrows():
        p: str = row["place"]
        d: str = row["dnm"]
        s: str = row["suff"]
        ruleKey: str = p[-2:]
        p = orthographicFix(p, s)
        ruleKey = getMostSimilarRule(ruleKey, rules)
        pred = ""
        if ruleKey in rules:
            pred = rules[ruleKey]
        if(pred == "" or s == "" or pd.isna(s)):
            outlier = outlier+1
        if(p+pred == d):
            acc += 1/data.__len__()
        else:
            print("failed for ", row["place"], "d:",
                  d, "s:", s, "p:", p, "pred:", pred)
            # if(pred == s and s.__len__() > 0):
            #     print("failed for ", row["place"], d, s, p, pred)
            #     pass
    print("OUTLIERS", outlier)
    return acc


def predictSingle(rules: dict, name:str):
    acc = 0
    outlier = 0
    p: str = name
    # d: str = row["dnm"]
    # s: str = row["suff"]
    ruleKey: str = p[-2:]

    # p = orthographicFix(p, s)
    ruleKey = getMostSimilarRule(ruleKey, rules)
    pred = ""
    if ruleKey in rules:
        pred = rules[ruleKey]
    ret = orthographicFix(p, pred)
    return {
        "suffix": pred,
       "full": ret+pred,
    }

rules = {}
pL2Dict = getPL2Hist(data)

pL2DictDF = pd.DataFrame.from_dict(pL2Dict)
for parts in pL2Dict:
    rules[parts] = max(pL2Dict[parts], key=pL2Dict[parts].get)

rulesData = {"2-Suffix": [], "Rule": []}
for key in rules:
    rulesData["2-Suffix"].append(key)
    rulesData["Rule"].append(rules[key])


st.title("Demonym generation from Place Name")


placeName = st.text_input('Enter place name', 'India')
if(placeName.strip()):
    st.write("The prediction is", predictSingle(rules, placeName))
