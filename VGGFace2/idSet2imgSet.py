# This script expand an id set to an image set linking each id to images
import os


def id2imgs(fileName, fileDestName):
    file = open(fileName, "r")
    fileDst = open(fileDestName, "w")
    txt = file.read()
    path = "C:\\Users\\Cuissot\\PycharmProjects\\Data\\VGGFacesV2\\train"
    outTxt = ""
    print(txt.split("\n"))
    for i in txt.split("\n"):
        print(i)
        images = os.listdir(path + "\\" + i)
        for j in images:
            outTxt += i + "\\" + j + "\n"
    fileDst.write(outTxt)

    fileDst.close()
    file.close()


def getFileElementsList(file):
    filecontent = open(file, "r")
    contentStr = filecontent.read()
    filecontent.close()
    list = contentStr.split('\n')
    return list


# just to check if we did the job correctly
def checker():
    idTrainList = getFileElementsList('Data/labels/homogeneousTrainSetIds.txt')
    id2check = getFileElementsList('Data/labels/homogeneousTrainImgs.txt')
    n = len(id2check)
    id = ""
    List = []
    for i in id2check:
        if i.split("\\")[0] != id:
            id = i.split("\\")[0]
            List.append(id)
    print(len(List))
    print(len(idTrainList))
    for i in range(len(List)):
        if List[i] != idTrainList[i]:
            print(List[i] + " " + idTrainList[i])


# id2imgs("Data/labels/homogeneousTrainSetIds.txt", "Data/labels/homogeneousTrainImgs.txt")
checker()
