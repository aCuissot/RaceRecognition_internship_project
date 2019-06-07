file1 = open("Data/labels/homogeneousTestLabels.txt", "r")
file2 = open("Data/labels/homogeneousTestImgs.txt", "r")
file3 = open("Data/labels/homogeneousTestSetIds.txt", "r")
file1_txt = file1.read()
file2_txt = file2.read()
file3_txt = file3.read()
labels_list = file1_txt.split("\n")
img_list = file2_txt.split("\n")
ids_list = file3_txt.split("\n")

finalTxt = ""
for i in img_list:
    id = i.split("\\")[0]
    finalTxt += i + ', ' + labels_list[ids_list.index(id)] + '\n'
# print(len(labels_list))
# print(len(img_list))
# print(finalTxt)
outputfile = open("Data/labels/homogeneousCsvTest.csv", "w")
outputfile.write(finalTxt)
outputfile.close()
file1.close()
file2.close()
file3.close()
