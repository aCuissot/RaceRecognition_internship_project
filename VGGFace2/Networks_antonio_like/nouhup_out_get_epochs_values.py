import matplotlib.pyplot as plt

f = open("data/nohup.out_resnet.txt", "r")
acc = 0
for x in f:
    if x.__contains__("31250/31250"):
        print(x)
        acc += 1
        print("Epoch ", acc)

# for resnet
loss = [0.4769, 0.3351, 0.2807, 0.2435, 0.2225, 0.2115, 0.1964, 0.1896, 0.1853, 0.1818, 0.1790, 0.1759, 0.1723, 0.171,
        0.1696, 0.1684]
acc = [0.8528, 0.8887, 0.9044, 0.9157, 0.9226, 0.9270, 0.9312, 0.9335, 0.9348, 0.9361, 0.9368, 0.9381, 0.9390, 0.9396,
       0.9401, 0.9406]
val_loss = [0.3388, 0.3234, 0.2613, 0.2362, 0.2534, 0.2263, 0.2183, 0.22, 0.2163, 0.2248, 0.2183, 0.2134, 0.2161,
            0.2159, 0.2149, 0.2173]
val_acc = [0.8857, 0.8919, 0.9124, 0.9206, 0.9228, 0.9256, 0.9282, 0.9273, 0.9293, 0.9272, 0.9296, 0.9315, 0.9304,
           0.9311, 0.9312, 0.9319]

# for vgg16
# loss = [0.3468, 0.2391, 0.2146, 0.1999, 0.1906, 0.1860, 0.1733, 0.1707, 0.1678, 0.1668, 0.1656, 0.1624, 0.1594, 0.1598,
#         0.1588, 0.1578, 0.157]
# acc = [0.8846, 0.9183, 0.9266, 0.9310, 0.9342, 0.9361, 0.94, 0.9409, 0.9419, 0.9425, 0.9429, 0.9436, 0.9447, 0.9451,
#        0.9454, 0.9453, 0.9457]
# val_loss = [0.2505, 0.2148, 0.2245, 0.2333, 0.2208, 0.2056, 0.2061, 0.2115, 0.224, 0.2128, 0.2135, 0.2104, 0.2076,
#             0.2084, 0.2156, 0.2183, 0.2124]
# val_acc = [0.9181, 0.93, 0.9305, 0.9328, 0.936, 0.9369, 0.9366, 0.9378, 0.9368, 0.9381, 0.9378, 0.9389, 0.9383,
#            0.9388, 0.9383, 0.9381, 0.9381]

fig, ax = plt.subplots()
ax.plot(loss, 'r', label='loss')
ax.plot(acc, 'g', label='acc')
ax.plot(val_loss, 'r--', label='val_loss')
ax.plot(val_acc, 'g--', label='val_acc')
plt.xlabel("Epochs")
legend = ax.legend(loc='center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')

plt.show()
# plt.plot(loss, 'r')
# plt.plot(acc, 'b')
# plt.plot(val_loss, 'k')
# plt.plot(val_acc, 'g')
# plt.show()
