import numpy as np
import matplotlib.pyplot as plt

dl = ['MPNN', 'LSTM', 'MPNN_LSTM']

f = open('../results/dep_results_FR_vacFalse_2020-12-27_2021-06-27.csv')

lines = f.readlines()
f.close()
lines = lines[2:]
asdsa
res1 = {}
res1_l = {}
res1_st = {}
dl_res1_l = {}
dl_res1_st = {}
for line in lines:
    l_spl = line.split(",")
    model = str(l_spl[0])
    nu = float(l_spl[2])
    st = float(l_spl[3])
    if model in res1:
        res1[model] +=  nu
        if model in dl:
            dl_res1_l[model].append(nu)
            dl_res1_st[model].append(st)
        else:
            res1_l[model].append(nu)
            res1_st[model].append(st)
    else:
        res1[model] = nu
        if model in dl:
            dl_res1_l[model] = [nu]
            dl_res1_st[model] = [st]
        else:
            res1_l[model] = [nu]
            res1_st[model] = [st]

for r in res1:
    print(r+" "+str(float(res1[r])/14))


f = open('../results/cum_pop_dep_results_FR_vacTrue_2020-12-27_2021-03-27.csv')
lines = f.readlines()
f.close()
lines = lines[4:]

res2 = {}
res2_l = {}
res2_st = {}
dl_res2_l = {}
dl_res2_st = {}
for line in lines:
    if "Mean" in line:
        continue
    l_spl = line.split(",")
    # print(l_spl)
    model = str(l_spl[0])
    nu = float(l_spl[2])
    st = float(l_spl[3])
    if model in res2:
        res2[model] +=  nu
        if model in dl:
            dl_res2_l[model].append(nu)
            dl_res2_st[model].append(st)
        else:
            res2_l[model].append(nu)
            res2_st[model].append(st)
    else:
        res2[model] = nu
        if model in dl:
            dl_res2_l[model] = [nu]
            dl_res2_st[model] = [st]
        else:
            res2_l[model] = [nu]
            res2_st[model] = [st]

for r in res2:
    print(r+" "+str(float(res2[r])/14))

# fig = plt.figure()
# x = np.arange(14)
# # y = 2.5 * np.sin(x / 20 * np.pi)
# yerr = np.linspace(0.05, 0.2, 10)
#
# for r in res1:
#     plt.errorbar(x, res1_l[r], yerr=res1_st[r], label=r)
#
# for r in res2:
#     plt.errorbar(x, res2_l[r], yerr=res2_st[r], label=r)
#
#
# # plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')
# #
# # plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
# #              label='uplims=True, lolims=True')
#
# upperlimits = [True, False] * 5
# lowerlimits = [False, True] * 5
# # plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
# #              label='subsets of uplims and lolims')
#
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig, axs = plt.subplots(4)
fig.tight_layout()
x = np.arange(14)

counter = 0
for r in res2_l:
    axs[counter].errorbar(x, res2_l[r], yerr=res2_st[r], label=r, color=cols[counter])
    for i, txt in enumerate(res2_l[r]):
        axs[counter].annotate(str(round(txt,1)), (x[i], txt+1000), ha='center', fontsize=6)
    axs[counter].set_ylim(0, 20000)
    axs[counter].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    counter += 1

fig.text(0.5, 0.02, 'Days', va='center', ha='center')
fig.text(-0.04, 0.5, 'Prediction error', va='center', ha='center', rotation='vertical')
fig.savefig('results_base.png', bbox_inches='tight', dpi=400)
plt.show()


cols = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']
## dl
fig, axs = plt.subplots(6)
fig.tight_layout()
x = np.arange(14)

counter = 0
for r in dl_res1_l:
    axs[counter].errorbar(x, dl_res1_l[r], yerr=dl_res1_st[r], label=r, color=cols[counter])
    for i, txt in enumerate(dl_res1_l[r]):
        axs[counter].annotate(str(round(txt,1)), (x[i], txt+80), ha='center', fontsize=8)
    axs[counter].set_ylim(0, 350)
    axs[counter].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    counter += 1

for r in dl_res2_l:
    axs[counter].errorbar(x, dl_res2_l[r], yerr=dl_res2_st[r], label=r+" (vac)", color=cols[counter])
    for i, txt in enumerate(dl_res2_l[r]):
        axs[counter].annotate(str(round(txt,1)), (x[i], txt+80), ha='center', fontsize=8)
    axs[counter].set_ylim(0, 350)
    axs[counter].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    counter += 1

fig.text(0.5, 0.02, 'Days', va='center', ha='center')
fig.text(-0.04, 0.5, 'Prediction error', va='center', ha='center', rotation='vertical')
fig.savefig('results.png', bbox_inches='tight', dpi=400)
plt.show()
