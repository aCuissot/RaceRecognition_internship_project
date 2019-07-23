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


def getEthIndex(line, ):
    """
    :param line: a line in the csv file
    :return: the index corresponding to this line
    """
    id = line.split(",")[0]
    index = sortedId.index(id)
    return index
