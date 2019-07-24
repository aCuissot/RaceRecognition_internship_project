a = open("Data/labels/labs_train.txt", "r")

txt = a.read()
ids = []
eth = []
beg_index = 0
index = 0
for i in txt:
    index += 1
    if i == "\n":
        eth.append(int(txt[beg_index:index - 1]) + 1)
        beg_index = index
    if i == "-":
        ids.append(txt[beg_index:index - 2])
        beg_index = index + 1
print(ids)
print(eth)
print(len(eth))
print(len(ids))

other_file = open("Data/labels/finalTrain.xml", "r")
final_txt = ""
index = 0
bool = False
for i in other_file:
    # print(i)
    if bool:
        i = "<ethnicity>" + str(eth[index]) + "</ethnicity>\n"
        index += 1
        bool = False
        print("change")
        print(i)
    if i.__contains__("id"):
        id = i.replace('<id>', '')
        id = id.replace('</id>\n', '')
        if id in ids:
            bool = True
    final_txt += i
# print(final_txt)
new_file = open("finalTrain.xml", "w")
new_file.write(final_txt)
new_file.close()
