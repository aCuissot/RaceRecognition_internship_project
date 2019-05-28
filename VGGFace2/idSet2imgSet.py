import os


def aaaaaaaaa(fileName, fileDestName):
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


aaaaaaaaa("Data/labels/homogeneousTrainSetIds.txt", "Data/labels/homogeneousTrainImgs.txt")
