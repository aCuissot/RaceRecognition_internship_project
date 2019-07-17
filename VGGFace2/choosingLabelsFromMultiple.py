fileLabel1 = open("")
fileLabel2 = open("")
fileLabel3 = open("")
txtLabel1 = fileLabel1.read()
txtLabel2 = fileLabel2.read()
txtLabel3 = fileLabel3.read()


def getId(list):
    sublist = []
    n = len(list)
    for i in range(0, n, 2):
        sublist.append(list[i])
    return sublist


def getEthnicity(list):
    sublist = []
    n = len(list)
    for i in range(1, n, 2):
        sublist.append(list[i])
    return sublist


def parseXML(xmlStr):
    xmlStr = xmlStr.replace("<xml>\n", "")
    xmlStr = xmlStr.replace("</xml>", "")
    xmlStr = xmlStr.replace("<subject>\n", "")
    xmlStr = xmlStr.replace("</subject>\n", "")

    xmlStr = xmlStr.replace("<curr_id>", "")
    xmlStr = xmlStr.replace("<ethnicity>", "")
    xmlStr = xmlStr.replace("</curr_id>", "")
    xmlStr = xmlStr.replace("</ethnicity>", "")

    list = xmlStr.split("\n")
    idList = getId(list)
    ethnicity = getEthnicity(list)
    return idList, ethnicity


def writeCategoryInXML(file, name, cat):
    file.write("<subject>\n")

    file.write("<curr_id>")
    file.write(str(name))
    file.write("</curr_id>\n")

    file.write("<ethnicity>")
    file.write(str(cat))
    file.write("</ethnicity>\n")

    file.write("</subject>\n")


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
    return 6

globalLabel = open("finalLabels.xml", 'w')

txt = "<xml>"
for i in range(len(array1)):
    curr_id = idList[i]
    finalCategory = getFinalCategory(array1[i], array2[i], array3[i])
    writeCategoryInXML(globalLabel, curr_id, finalCategory)
