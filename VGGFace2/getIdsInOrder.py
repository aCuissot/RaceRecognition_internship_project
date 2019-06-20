ids_file = open("Data/labels/homogeneousTrainSetIds.txt", "r")
ids_txt = ids_file.read()
ids_txt = ids_txt.replace("n", "")
ids_list = ids_txt.split("\n")
ids_list = ids_list[8:]
ids_list.sort()
for i in ids_list:
    print(i)
