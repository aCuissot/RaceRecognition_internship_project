import io


def getId(list):
    """
    :param list: list of IDs and Ethnicities done by parseXML function
    :return: The ID list contained in the input list
    """
    sublist = []
    n = len(list)
    for i in range(0, n - 1, 2):
        sublist.append(list[i])
    return sublist


def getEthnicity(list):
    """
    :param list: list of IDs and Ethnicities done by parseXML function
    :return: The ethnicities list contained in the input list
    """
    sublist = []
    n = len(list)
    for i in range(1, n, 2):
        sublist.append(list[i])
    return sublist


def parseXML(xmlStr):
    """
    :param xmlStr: String containing the content of the XML file
    :return: the IDs and ethnicities lists of the XML file
    """
    xmlStr = xmlStr.replace("<xml>\n", "")
    xmlStr = xmlStr.replace("</xml>", "")
    xmlStr = xmlStr.replace("<subject>\n", "")
    xmlStr = xmlStr.replace("</subject>\n", "")

    xmlStr = xmlStr.replace("<curr_id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</curr_id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    id = getId(list)
    ethnicity = getEthnicity(list)
    return id, ethnicity


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


def getEthIndex(line, ):
    """
    :param line: a line in the csv file
    :return: the index corresponding to this line
    """
    id = line.split(",")[0]
    index = sortedId.index(id)
    return index


# Adding ethnicities data to the csv
header = csv.readline()
header = header[:len(header) - 1]
csvdst.write(header + ", Ethnicity\n")
i = 0

for line in csv:
    newLine = line[:len(line) - 1] + ", " + sortedEth[getEthIndex(line)] + "\n"
    csvdst.write(newLine)
    i += 1
