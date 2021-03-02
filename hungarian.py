from scipy.optimize import linear_sum_assignment
import numpy as np
import sys
from IPython import embed

np.set_printoptions(linewidth=150)
np.set_printoptions(threshold=sys.maxsize)

cost = np.array([
    [0.0152724,   20.9558,   57.2282,   309.663,   804.877,   327.016,   2477.19,   2550.79,    2726.2,   339.985,   780.909,   232.704,   2472.21,   426.568,   444.506,   3417.65,   1993.88,
        1820.93,   234.677,   2497.49,   2445.41,   2443.04,   2451.53,   2540.17,   2732.41,   3110.85,   3377.01,   2575.88,   486.685,   806.313,   3353.36,   4070.89,   2947.17,      2314,   2017.06],
    [20.9611, 0.00210027,   10.5103,   169.505,   709.412,   182.485,   2427.69,   2446.78,   2563.15,   462.513,    545.97,   357.896,   2519.44,   258.613,   416.416,    3102.8,   1625.02,   1451.41,
        289.82,   2562.03,   2452.23,   2421.31,   2402.21,   2437.49,   2567.63,   2863.22,   3007.24,   2191.02,   306.301,    854.32,   3068.45,   3671.07,   2535.98,   1917.52,   1627.28],
    [57.5394,   10.7241, 0.00171145,   107.067,     598.3,   113.849,   2274.22,   2261.12,   2342.17,   517.979,    426.02,    428.61,    2424.6,   178.912,   361.972,   2795.43,   1379.32,   1239.64,
        303.391,    2477.6,   2332.93,   2284.97,   2249.59,   2252.83,   2345.59,      2592,   2684.04,   1902.13,   221.863,   823.393,   2774.52,   3314.95,   2224.54,   1649.22,   1401.29],
    [326.551,    182.41,   114.474,   1.63506,   634.663, 0.00122571,   2452.55,   2312.13,   2255.48,   1044.76,   99.9403,   954.805,   2828.16,   8.94756,   547.332,   2352.49,   763.614,
        605.3,   673.801,   2921.51,   2642.41,   2527.82,      2428,   2306.85,   2254.89,   2307.85,   2104.53,   1265.83,   21.5644,   1197.19,   2403.64,   2672.77,   1530.22,   972.124,   720.749],
    [509.904,   325.685,   241.166,   27.2388,    856.87,   28.4075,   2860.34,   2664.33,    2548.4,   1412.88,   33.3142,   1298.28,   3331.63,   5.76418,   804.782,   2483.74,   652.764,   425.391,
        978.238,   3442.26,   3105.78,   2962.49,    2834.4,   2659.89,   2546.04,    2515.1,   2147.11,   1206.49,  0.531368,   1574.04,   2574.31,   2715.14,    1458.1,   846.029,   524.934],
    [308.908,    169.22,   107.576, 0.000693185,   689.857,   1.63654,   2559.67,   2423.73,    2372.3,   1051.47,   107.706,   950.319,   2926.14,   9.68049,   583.524,   2479.07,   819.561,
        633.826,   687.969,   3017.92,   2744.16,   2632.08,   2534.48,   2418.14,   2371.86,   2431.85,   2222.45,   1350.27,    21.238,   1244.03,   2531.27,   2805.36,   1622.46,   1035.48,   752.819],
    [253.274,   418.827,   549.321,   1116.99,   1553.62,   1153.79,   3155.48,   3415.09,   3792.61,   297.192,   1914.35,   163.784,    2818.5,   1328.72,   942.416,   4997.91,   3642.01,    3428.7,
        424.928,   2784.45,   2930.29,   3025.64,    3128.2,    3399.6,   3804.69,    4466.3,   5124.33,   4320.86,   1428.04,   1070.64,   4836.93,   5939.74,   4790.75,   4064.41,   3696.92],
    [177.508,   233.708,   253.229,   634.179,   433.666,   625.224,   1459.95,    1593.4,   1829.66,   56.1527,   1192.49,   63.3149,   1361.55,   780.334,   143.337,   2716.74,   2154.84,   2276.39,
        4.45018,   1369.44,   1374.19,   1401.53,    1440.7,   1583.39,   1837.67,    2304.7,   2877.35,   2463.92,   876.347,   240.687,   2582.05,    3473.8,   2803.86,   2450.36,   2474.72],
    [491.303,   541.942,   530.697,   935.155,   267.012,    904.05,   862.484,   998.836,   1235.35,   113.074,    1486.1,   204.523,   768.154,   1082.11,   76.6817,   2151.34,    2169.6,   2502.48,
        50.1789,   776.291,   779.419,   807.064,   848.003,   990.261,    1243.4,   1713.07,   2416.17,   2270.51,   1194.78,   39.9926,   1988.02,   2920.38,    2568.5,   2427.02,   2687.43],
    [444.686,   416.689,   361.622,    584.76,   79.0152,   549.024,   833.185,   865.569,   989.929,   301.929,    956.96,    384.46,   921.848,   678.802, 0.0361076,   1612.89,   1432.41,
        1729,   112.939,   962.819,   856.631,   830.002,   818.273,   859.289,     994.7,   1306.61,   1756.16,   1537.44,    765.11,   130.061,   1511.18,    2218.6,   1792.39,    1643.7,    1877.7],
    [805.42,   709.823,   598.665,   690.547, 0.000999248,   636.062,   591.418,   541.431,   574.988,   679.975,   912.608,   812.309,    831.27,   742.458,   78.9779,   979.861,   1030.72,   1440.84,
        381.03,   899.047,   703.094,   632.642,   579.399,   537.999,    577.14,   762.849,   1094.98,   998.188,   818.649,    253.66,   908.688,   1461.09,   1189.84,   1178.78,   1551.05],
    [780.598,   546.002,   427.458,   107.246,   911.766,   99.6053,   2898.35,    2638.8,   2453.54,   1761.12, 0.00499273,   1663.56,   3483.91,   54.4487,   955.149,   2215.52,   416.708,   223.264,
        1245.01,   3614.93,   3210.44,   3033.61,   2873.48,   2636.23,   2449.22,   2321.32,   1831.26,   917.088,   37.3425,   1792.29,   2336.31,    2348.7,   1130.83,    571.05,   296.671],
    [232.082,   357.346,   426.871,   951.413,   814.452,   955.907,   1891.43,    2111.7,   2442.85,   21.5152,   1664.51, 0.00354622,   1636.27,   1141.74,   386.862,   3565.68,    2922.8,
        2984.3,   81.5583,   1616.24,   1714.28,   1787.46,   1870.48,   2099.11,   2453.61,   3053.16,   3784.54,   3317.91,   1252.93,   400.448,   3391.18,   4456.05,   3712.34,   3273.81,   3219.12],
    [344.415,   466.858,   520.441,   1057.38,   677.025,   1050.36,   1529.77,   1751.65,   2083.04, 0.0218891,    1766.8,   22.3594,   1276.56,   1248.19,   301.339,   3221.19,   2880.26,   3051.71,
        54.3575,   1256.62,   1353.92,   1427.32,    1511.3,   1739.81,   2093.82,    2694.9,   3494.38,   3167.09,   1368.07,   245.055,   3031.74,   4117.94,   3539.74,    3211.5,   3279.61],
    [809.217,   857.174,   825.067,   1248.45,   254.621,   1202.09,   549.821,   689.667,    927.96,   249.853,   1796.98,   399.727,   455.689,   1396.53,   131.885,   1868.11,    2290.5,   2770.09,
        181.1,   463.528,   467.087,   495.731,   538.634,   682.041,   936.078,   1409.86,   2206.83,   2247.41,    1521.1, 0.00532337,   1684.16,   2647.85,      2517,   2522.27,   2946.33],
    [2496.57,    2560.6,   2474.67,   3017.88,   899.493,   2922.77,   155.444,   339.944,   619.858,   1266.08,   3614.93,   1611.19,   3.61153,   3187.35,   963.732,   1733.12,   3549.57,   4523.74,
        1217.9, 0.0089743,   39.8699,   89.0821,   155.289,   334.758,    629.28,   1166.83,   2353.66,   3030.84,   3359.51,   465.657,   1461.39,   2599.86,   3220.62,   3711.44,   4682.76],
    [2438.8,   2446.46,   2327.03,   2743.07,   704.064,   2642.95,   39.6324,   150.402,    350.56,   1355.99,   3212.37,   1701.27,   18.2821,   2874.99,   855.227,   1257.63,   2977.51,   3942.18,
        1217.24,   38.0546, 0.0318935,   10.6604,   39.5412,   146.962,   357.654,   783.177,   1799.73,   2445.62,   3029.68,   465.406,   1027.49,   2009.09,   2601.55,   3103.62,   4074.37],
    [2435.05,   2411.67,   2273.93,   2619.77,   626.832,      2517,   8.41132,    78.536,   234.741,    1434.9,   3018.71,   1779.85,    58.946,   2731.01,   825.401,   1028.03,   2691.82,   3650.21,
        1246.29,   91.5987,   10.6647, 0.0290369,   8.33216,   76.0509,   240.552,   604.329,    1526.4,   2153.72,   2875.99,   495.903,   820.963,   1715.91,    2291.2,   2798.41,   3767.69],
    [2476.24,   2426.75,   2272.81,   2560.65,   592.551,   2455.02, 0.00472523,   36.2383,    155.71,   1537.68,   2900.56,   1884.81,   110.748,   2654.63,   833.669,   854.744,    2488.6,   3447.91,
        1307.06,   154.174,   37.3231,   8.92139, 0.0753274,   34.5939,   160.451,    472.69,   1319.59,   1939.21,   2791.98,   550.682,   666.115,   1489.82,   2060.05,   2577.76,   3552.69],
    [2556.42,   2451.95,   2265.74,   2429.07,    544.79,   2318.85,   35.8071, 0.00573831,   41.5938,   1765.54,   2644.33,    2111.7,   273.907,   2486.41,   869.434,   538.895,   2070.62,
        3017.8,   1446.62,   340.156,   147.276,   81.1934,   35.8775, 0.0481558,   44.0623,   246.877,   925.737,   1511.19,   2606.48,   693.978,   391.264,   1061.24,   1599.54,   2125.37,   3096.57],
    [2728.38,   2564.77,   2343.72,   2373.31,   576.736,   2258.04,   155.387,   42.1513, 0.00336151,   2094.98,   2454.49,   2440.52,   530.436,   2391.27,   991.696,    280.56,   1705.46,   2640.67,
        1681.05,   621.224,   346.594,   239.999,   155.482,   44.0155, 0.0299273,   85.2081,   584.069,   1134.16,   2492.84,   931.171,   176.858,    681.75,   1187.37,   1723.01,    2691.4],
    [2918.49,   2710.96,   2463.79,   2394.37,   658.609,   2275.15,   298.358,   127.705,   23.4402,   2399.58,   2376.02,   2745.43,    775.04,   2383.13,   1141.48,   143.471,   1494.39,   2422.65,
        1915.07,   884.033,   548.743,   412.097,   298.516,   130.934,   21.6561,   19.5898,   386.104,   912.398,   2471.14,      1165,   72.0955,   454.891,   939.151,   1484.01,   2452.46],
    [3410.65,   3118.74,   2821.53,   2561.79,   944.323,   2435.07,   702.799,   421.826,   198.222,   3112.51,   2352.23,   3458.69,   1374.72,    2494.5,   1557.34,   9.76264,   1216.74,   2130.98,
        2492.42,   1518.69,   1066.87,   872.487,   703.064,   427.675,    192.96,   23.2699,    135.11,   614.848,   2556.38,   1742.77,  0.578127,   148.672,   590.908,    1152.8,   2120.63],
    [4048.4,   3651.88,    3299.5,   2794.84,   1442.75,   2664.21,   1457.97,   1037.69,   664.117,   4091.74,   2346.92,   4417.29,   2374.13,   2656.94,   2194.27,   80.9326,   962.205,   1806.04,
        3304.14,   2561.97,   1964.17,   1697.19,   1457.18,   1046.47,   654.477,   273.565,   32.9159,   387.293,   2681.85,   2616.56,   157.015,  0.146821,   310.154,   840.484,   1748.59],
    [2960.37,   2542.47,   2232.54,   1607.15,   1241.08,   1518.16,   2168.41,   1699.71,   1279.54,   3605.45,   1100.26,   3761.64,   3151.78,    1445.9,   1841.79,   519.085,   178.194,   591.684,
        2788.03,   3352.62,    2713.1,   2423.19,   2158.07,   1706.61,   1268.47,     817.5,   216.219,   14.1887,   1427.24,   2596.08,   676.757,   347.613,   1.83657,   112.396,   545.626],
    [2396.35,   1992.64,   1721.57,   1088.16,   1229.94,   1024.47,    2624.2,   2162.75,   1753.62,   3305.58,   607.622,   3368.19,   3584.51,    931.31,   1707.31,    990.36,   19.1607,   199.411,
        2513.47,   3781.76,    3155.1,   2870.33,   2608.55,   2167.81,   1742.84,   1304.84,   582.945,   88.1129,   896.633,      2595,   1173.91,   821.972,   127.142,  0.733692,   172.018],
    [2092.67,   1695.78,   1467.38,   798.701,    1598.9,   766.044,   3599.88,   3133.06,   2722.18,   3365.43,   326.251,   3309.61,   4556.76,   640.092,   1937.29,   1914.68,   126.631,   9.77903,
        2587.44,   4754.07,   4128.34,   3842.45,   3577.53,   3136.13,   2711.29,   2267.61,   1359.44,    487.83,   581.703,   3015.23,   2139.18,   1727.43,   585.211,   161.091,  0.707981],
])

print ("original not square")
print ("cost: {} detections X {} tracks".format(cost.shape[0], cost.shape[1]))

row_ind, col_ind = linear_sum_assignment(cost)

# print(row_ind)
# print(col_ind)
# print (cost[row_ind, col_ind])

result = -np.ones(cost.shape)
result[row_ind, col_ind] = 0.
print (result)

print ("square")
c = np.pad(cost,  ((0, 8),  (0,  0)), 'constant')
print ("cost: {} detections X {} tracks".format(c.shape[0], c.shape[1]))

row_ind, col_ind = linear_sum_assignment(c)

result = -np.ones(c.shape)
result[row_ind, col_ind] = 0.
print (result)

print ("max cost")
c = cost
c[c > 100] = 100

print ("cost: {} detections X {} tracks".format(c.shape[0], c.shape[1]))
row_ind, col_ind = linear_sum_assignment(c)

result = -np.ones(c.shape)
result[row_ind, col_ind] = 0.
print (result)

print("-----------------------------------------------------------------------------------")
print ("original 5 by 4")
cost = np.array([
    [18, 11, 16, 20],
    [14, 19, 26, 18],
    [21, 23, 35, 29],
    [32, 27, 21, 17],
    [16, 15, 28, 25]
])

print ("cost: {} detections X {} tracks".format(cost.shape[0], cost.shape[1]))

row_ind, col_ind = linear_sum_assignment(cost)

result = -np.ones(cost.shape)
result[row_ind, col_ind] = 0.
print (result)

print ("squared by adding 0")
cost = np.array([
    [18, 11, 16, 20, 0],
    [14, 19, 26, 18, 0],
    [21, 23, 35, 29, 0],
    [32, 27, 21, 17, 0],
    [16, 15, 28, 25, 0]
])

print ("cost: {} detections X {} tracks".format(cost.shape[0], cost.shape[1]))

row_ind, col_ind = linear_sum_assignment(cost)

result = -np.ones(cost.shape)
result[row_ind, col_ind] = 0.
print (result)

print ("squared by adding max value")
cost = np.array([
    [18, 11, 16, 20, 35],
    [14, 19, 26, 18, 35],
    [21, 23, 35, 29, 35],
    [32, 27, 21, 17, 35],
    [16, 15, 28, 25, 35]
])

print ("cost: {} detections X {} tracks".format(cost.shape[0], cost.shape[1]))

row_ind, col_ind = linear_sum_assignment(cost)

result = -np.ones(cost.shape)
result[row_ind, col_ind] = 0.
print (result)
