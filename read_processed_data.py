import re
import matplotlib.pyplot as plt

f1 = open('./savepoint/records_cross5_1.txt', 'r')
f2 = open('./savepoint/records_cross5_2.txt', 'r')
f3 = open('./savepoint/records_cross5_3.txt', 'r')

train_list1 = []
train_list2 = []
train_list3 = []

# traj loss
for line in f1.readlines():
    line = line.strip()
    print(line, len(line))

    for a in re.finditer(':', line):
        # print(a.span())
        figure = float(line[a.span()[1]:])
        train_list1.append(figure)
# endpoint loss
for line in f2.readlines():
    line = line.strip()
    print(line, len(line))

    for a in re.finditer(':', line):
        # print(a.span())
        figure = float(line[a.span()[1]:])
        train_list2.append(figure)

# discriminator loss
for line in f3.readlines():
    line = line.strip()
    print(line, len(line))

    for a in re.finditer(':', line):
        # print(a.span())
        figure = float(line[a.span()[1]:])
        train_list3.append(figure)

print(len(train_list1), train_list1)
print(len(train_list2), train_list2)
print(len(train_list3), train_list3)

x = []
for i in range(len(train_list1)):
    x.append(i)
print(x)

plt.plot(x, train_list3, linewidth=3)

plt.xlabel("train epoch")
plt.ylabel("discriminator loss")
plt.show()