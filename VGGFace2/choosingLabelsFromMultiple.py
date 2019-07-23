# We will label 2 more times to avoid biased judgement on ethnicity, this script is to merge resulting XMLs
from VGGFace2.utils.xmlParserFnct import *

fileLabel1 = open("Data/labels/testP.xml")
fileLabel2 = open("Data/labels/test1.xml")
fileLabel3 = open("Data/labels/test2.xml")
txtLabel1 = fileLabel1.read()
txtLabel2 = fileLabel2.read()
txtLabel3 = fileLabel3.read()

idList, array1 = parseXML(txtLabel1)
idList, array2 = parseXML(txtLabel2)
idList, array3 = parseXML(txtLabel3)

if (len(array1) != len(array2)) or (len(array1) != len(array3)) or (len(array2) != len(array3)):
    print("lens are not matching")


def getFinalCategory(param, param1, param2):
    if param == param1:
        return param
    if param == param2:
        return param
    if param2 == param1:
        return param1
    # if there is 3 differents choices, we will decide later so 6 is a temporary category
    return 6


globalLabel = open("finalLabels.xml", 'w')
globalLabel.write("<xml>")
for i in range(len(array1)):
    curr_id = idList[i]
    finalCategory = getFinalCategory(array1[i], array2[i], array3[i])
    writeCategoryInXML(globalLabel, curr_id, finalCategory)
globalLabel.write("</xml>")

globalLabel.close()
