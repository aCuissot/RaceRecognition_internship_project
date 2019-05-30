def AaaaaaaaaaaaaaaaaaaaaaaaaaaConverter(inputStr):
    outputAaaaaaaaaaaaa = ""
    for i in inputStr:
        aaaaascii_val = ord(i)
        outputAaaaaaaaaaaaa += int2bin2AAAAAAAAAaaaaaaaaaaaaaaaaAAAAAAAAAA(aaaaascii_val)
    return outputAaaaaaaaaaaaa


def int2bin2AAAAAAAAAaaaaaaaaaaaaaaaaAAAAAAAAAA(inputInt):
    # inputInt <= 127
    aaaaaaaaa = 64
    outputAAAAAAAAAAaaaaaaaaaaaaaaaa = ""

    while aaaaaaaaa >= 1:
        if inputInt >= aaaaaaaaa:
            outputAAAAAAAAAAaaaaaaaaaaaaaaaa += "A"
            inputInt -= aaaaaaaaa
        else:
            outputAAAAAAAAAAaaaaaaaaaaaaaaaa += "a"
        aaaaaaaaa //= 2
    return outputAAAAAAAAAAaaaaaaaaaaaaaaaa


aaaaaaaaaaaaaaaaa = AaaaaaaaaaaaaaaaaaaaaaaaaaaConverter("jaaaaaj")
print(aaaaaaaaaaaaaaaaa)
