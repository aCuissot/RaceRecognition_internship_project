import io
from VGGFace2.utils.xmlParserFnct import *


# Loading xml files
train = open("Data/labels/TrainXML.xml", "r")
trainXML = train.read()
TrainIdList, TrainEthList = parseXML(trainXML)
test = open("Data/labels/TestXML.xml", "r")
testXML = test.read()
TestIdList, TestEthList = parseXML(testXML)

# Concatenate Ids lists, concatenate ethnicities lists
FullIdList = TrainIdList.copy()
FullIdList += TestIdList

FullEthList = TrainEthList.copy()
FullEthList += TestEthList

FullList = list(zip(FullIdList, FullEthList))
FullList.sort()
test.close()
train.close()
print(len(FullList))
sortedId, sortedEth = zip(*FullList)
print(sortedEth)

# Opening csv files
csv = io.open("Data/labels/identity_meta.csv", "r", encoding="utf-8")
csvdst = io.open("Data/labels/new_identity_meta.csv", "w", encoding="utf-8")

# Adding ethnicities data to the csv
header = csv.readline()
header = header[:len(header) - 1]
csvdst.write(header + ", Ethnicity\n")
i = 0

for line in csv:
    newLine = line[:len(line) - 1] + ", " + sortedEth[getEthIndex(line, sortedId)] + "\n"
    csvdst.write(newLine)
    i += 1
