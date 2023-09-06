import numpy as np

f = open('../results/dep_results_FR_vacTrue_2020-12-27_2021-03-27.csv','r')
lines = f.readlines()
f.close()
lines = lines[2:]
res = {}
for line in lines:
    l = line.split(",")
    model = l[0]
    err = float(l[2])

    model[res].append(err)


for m in res:
    print(m+" "+res[m].)

f = open('../results/dep_results_FR_vacFalse_2020-12-27_2021-03-27.csv','r')
lines = f.readlines()
f.close()
