#!/usr/bin/env python3
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os

import rosbag
import rospy

folder = "/home/adi/Projects/traffic/ground_removal/cetran-beae5851e_heightmap+ring+minhits1_2021-06-08-17-53-31"

blocked_topic = '/navigation/dynamic_path_checker/blocked'
vel_topic = '/donner_articulated_drive_controller/odom'
cmdvel_topic = '/cmd_vel'

SHOW = False
SAVE = True
PLOT_SPEED = False

files = glob(os.path.join(folder, '*.bag'))

for file in files:
    bag = rosbag.Bag(file, 'r')
    filename = os.path.basename(file)

    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    total_time = end_time - start_time

    time_list = list()
    blocked_data = list()
    vel_time_list = list()
    vel_list = list()
    cmdvel_time_list = list()
    cmdvel_list = list()

    blocked_msgs_count = 0
    total_msgs_count = 0
    last_msg_free = False
    last_free_timestamp = None
    free_time = rospy.Duration(0, 0)

    for topic, message, time in bag.read_messages([blocked_topic, vel_topic, cmdvel_topic]):
        if topic == blocked_topic:
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

        elif topic == vel_topic:
            vel_time_list.append(time.to_sec())
            vel_list.append(message.twist.twist.linear.x)

        elif topic == cmdvel_topic:
            cmdvel_time_list.append(time.to_sec())
            cmdvel_list.append(message.drive.speed)

    bag.close()

    print(f"---- {filename} ----")

    if total_msgs_count == 0 or len(blocked_data) == 0:
        print("no blocked data")
        continue

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
    vel_list = np.array(vel_list).astype(np.float)
    vel_time_list = np.array(vel_time_list).astype(np.float)
    cmdvel_list = np.array(cmdvel_list).astype(np.float)
    cmdvel_time_list = np.array(cmdvel_time_list).astype(np.float)

    f = plt.figure(figsize=(14, 9))
    plt.suptitle(filename)
    # plt.title(f"free time: {free_time_percentage:.3f}%, blocked msgs: {blocked_msgs_percentage:.3f}%")
    plt.title(f"free time: {free_time_percentage:.3f}%")
    plt.plot(time_list - time_list[0], blocked_data, linestyle='--',
             linewidth=1, marker='.', color='b', label='blocked')
    if PLOT_SPEED:
        plt.plot(vel_time_list - vel_time_list[0], vel_list, linestyle='--',
                 linewidth=1, markersize=2, marker='.', color='r', label='vel')
        plt.plot(cmdvel_time_list - cmdvel_time_list[0], cmdvel_list, linestyle='--',
                 linewidth=1, markersize=2, marker='.', color='g', label='cmdvel')
    # plt.plot(time_list - time_list[0], blocked_bool, linestyle='-',
    #          linewidth=1, marker='.', color='r', label='bool')
    # plt.ylim([-2, 45])
    plt.xlim([-5, 175])
    # plt.xlim([-5, 590])
    plt.legend()
    plt.grid()

    # f.tight_layout()
    if SAVE:
        plt.savefig(os.path.splitext(file)[0] + ".png")

if SHOW:
    plt.show()
print("done")
