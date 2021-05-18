from collections import defaultdict
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "/home/adi/Projects/traffic/Kalman_Filter_tuning/"

min_N = 10
y_scale = 10
SHOW = True
SAVE = True

files = glob(os.path.join(folder, '*.log'))
for file in files:
    with open(file) as f:
        f = f.readlines()

    ids = set()
    score_all = defaultdict(list)
    nis_avg_all = defaultdict(list)
    S_all = defaultdict(list)
    y_all = defaultdict(list)

    for i in range(len(f)):
        # for line in f:
        line = f[i]
        if "track id" in line:
            line = line.replace('track id: ', '')
            end = line.find(', cluster')
            line = line[:end]
            id = int(line)

            # nis metric
            line = f[i+4]
            if "NIS score" not in line:
                # print("Error! NIS score should be after track id")
                continue

            id_start = line.find(': ') + 2
            id_end = line.find(' : ')
            new_id = int(line[id_start:id_end])

            if (new_id != id):
                # print("different IDs")
                # TODO handle different IDs
                continue

            ids.add(id)
            # get measurements
            nis_avg_start = line.find(' : ') + 3
            nis_avg_end = line.find(' ( ')
            nis_avg = float(line[nis_avg_start:nis_avg_end])

            N_start = line.find(' ( ') + 3
            N_end = line.find(' ) = ')
            N = int(line[N_start:N_end])
            # print(N)

            NIS_start = line.find(' = ') + 3
            NIS_end = line.find('  =>')
            NIS = float(line[NIS_start:NIS_end])
            # print(NIS)

            score_start = line.find('=>') + 4
            score_end = -1
            score = float(line[score_start:score_end])

            if (N == min_N):
                score_all[id].append(score)
                nis_avg_all[id].append(nis_avg * N)

            # S
            line = f[i+1]
            if "Sx" not in line:
                # print("Error! Sx should be after track id")
                continue

            line = line.replace('Sx: ', '')
            line = line.replace('\n', '')
            S = float(line)

            if (N == min_N):
                S_all[id].append(S)

            # Y
            line = f[i+2]
            if "y" not in line:
                # print("Error! Sx should be after track id")
                continue

            line = line.replace('y: ', '')
            line = line.replace('\n', '')
            y = np.array(np.matrix(line)).ravel()
            y = np.linalg.norm(y)

            if (N == min_N):
                y_all[id].append(y * y_scale)

            # nis
            line = f[i+3]
            if "nis" not in line:
                # print("Error! nis should be after track id")
                continue

            line = line.replace('nis: ', '')
            line = line.replace('\n', '')
            nis = float(line)

    for id in ids:
        # print(id)
        score_1 = score_all[id].count(1)
        percentage_score = 100.0 * score_1 / len(score_all[id])
        # print(f"{percentage_score:.2f}")

        f = plt.figure()
        filename = os.path.basename(file)
        plt.suptitle(filename)
        plt.title(f"id: {id} , score: {percentage_score:.2f}%")
        plt.plot(nis_avg_all[id], 'g', label='nis avg')
        plt.plot(S_all[id], 'b', label='S')
        plt.plot(y_all[id], 'y', label='Y norm')
        plt.plot(score_all[id], 'r', label='score')
        plt.axhline(y=20.483, color='k', linestyle='--')
        plt.axhline(y=3.247, color='k', linestyle='--')
        plt.ylim([-2, 45])
        plt.legend()
        plt.grid()

        if SAVE:
            plt.savefig(os.path.splitext(file)[0] + ".png")

    if SHOW:
        plt.show()
