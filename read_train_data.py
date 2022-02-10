import matplotlib
import re

file = open('./savepoint/records_cross5.txt', 'r')
f1 = open('./savepoint/records_cross5_1.txt', 'w')
f2 = open('./savepoint/records_cross5_2.txt', 'w')
f3 = open('./savepoint/records_cross5_3.txt', 'w')
i = 0
discrim = 0
for line in file.readlines():
    line = line.strip()
    print(line, len(line))
    i += 1
    if len(line) < 60:
        line3 = line
        f3.write(line3)
        f3.write('\n')
    else:
        for a in re.finditer('E', line):
            # print(a.span())
            b = a.span()[0]
            if b != 0:
                line1 = line[:b]
                line2 = line[b:]
                f1.write(line1)
                f1.write('\n')
                f2.write(line2)
                f2.write('\n')
                print("line1", line1)
                print("line2", line2)

    '''
    if line[11] == ' ':
    # 轮数大于100
    if line[9] == ' ':
        if line[10] == 'd':

        elif line[10] == 'w':

    if line[10] == ' ':
        if line[11] == 'd':

        elif line[11] == 'w':
    '''
file.close()
f1.close()
f2.close()
f3.close()