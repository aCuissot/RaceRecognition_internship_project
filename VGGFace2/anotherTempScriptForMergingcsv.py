file1 = open("Data/labels/homogeneousTrainLabels.txt", "r")
file2 = open("Data/labels/homogeneousTrainImgs.txt", "r")
file3 = open("Data/labels/homogeneousTrainSetIds.txt", "r")
file1_txt = file1.read()
file2_txt = file2.read()
file3_txt = file3.read()
labels_list = file1_txt.split("\n")
img_list = file2_txt.split("\n")
ids_list = file3_txt.split("\n")

finalTxt = ""
for i in range(len(ids_list)):
    id = ids_list[i]
    for img in img_list:
        if img.split("\\")[0] == id:
            finalTxt += img + ', ' + labels_list[i] + '\n'
# print(len(labels_list))
# print(len(img_list))
# print(finalTxt)
outputfile = open("Data/labels/homogeneousCsvTrain.csv", "w")
outputfile.write(finalTxt)
outputfile.close()
file1.close()
file2.close()
file3.close()

