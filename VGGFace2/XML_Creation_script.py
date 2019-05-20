def getId(id):
    if i < 10:
        return "n00000" + str(id)
    if i < 100:
        return "n0000" + str(id)
    if i < 1000:
        return "n000" + str(id)
    else:
        return "n00" + str(id)


XML = "<base>\n"
for i in range(1, 9280):
    XML += "<subject>\n"
    XML += "<id>" + getId(i) + "</id>\n"
    XML += "<ethnicity></ethnicity>\n"
    XML += "</subject>\n"
XML += "</base>\n"

f = open("XML.xml", "w")
f.write(XML)
f.close()
