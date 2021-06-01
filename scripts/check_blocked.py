#!/usr/bin/env python3
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import rosbag
import rospy

folder = "/home/adi/Projects/traffic/ground_removal/"
topic = '/navigation/dynamic_path_checker/blocked'

SHOW = True
SAVE = True

files = glob(os.path.join(folder, '*.bag'))

for file in files:
    bag = rosbag.Bag(file, 'r')
    filename = os.path.basename(file)

    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    total_time = end_time - start_time

    time_list = list()
    blocked_data = list()
    # blocked_bool = list()

    blocked_msgs_count = 0
    total_msgs_count = 0
    last_msg_free = False
    last_free_timestamp = None
    free_time = rospy.Duration(0, 0)

    for _, message, time in bag.read_messages(topic):
        time_list.append(time.to_sec())
        blocked_data.append(message.data)
        total_msgs_count += 1

        if message.data > -1.0:
            blocked_msgs_count += 1
            if last_msg_free:
                free_time += (time - last_free_timestamp)
            last_msg_free = False
            # blocked_bool.append(1)
        else:
            last_msg_free = True
            last_free_timestamp = time
            # blocked_bool.append(0)

    bag.close()

    print(f"---- {filename} ----")
    # print(f"start time : {start_time}")
    # print(f"end time : {end_time}")
    # print(f"total_time : {total_time}")

    # print(f"blocked_msgs_count : {blocked_msgs_count}")
    # print(f"total_msgs_count : {total_msgs_count}")
    print(f"free_time : {free_time.to_sec()}")

    free_time_percentage = 100.0 * free_time.to_sec() / total_time
    blocked_msgs_percentage = 100.0 * blocked_msgs_count / total_msgs_count

    blocked_data = np.array(blocked_data).astype(np.float)
    time_list = np.array(time_list).astype(np.float)

    f = plt.figure(figsize=(14, 9))
    plt.suptitle(filename)
    # plt.title(f"free time: {free_time_percentage:.3f}%, blocked msgs: {blocked_msgs_percentage:.3f}%")
    plt.title(f"free time: {free_time_percentage:.3f}%")
    plt.plot(time_list - time_list[0], blocked_data, linestyle='--',
             linewidth=1, marker='.', color='b', label='blocked')
    # plt.plot(time_list - time_list[0], blocked_bool, linestyle='-',
    #          linewidth=1, marker='.', color='r', label='bool')
    # plt.ylim([-2, 45])
    plt.xlim([-5, 140])
    plt.legend()
    plt.grid()

    # f.tight_layout()
    if SAVE:
        plt.savefig(os.path.splitext(file)[0] + ".png")

if SHOW:
    plt.show()
print("done")
