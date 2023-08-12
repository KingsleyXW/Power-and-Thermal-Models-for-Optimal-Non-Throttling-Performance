import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def read_json(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as file_obj:
        return json.load(file_obj)

def read_csv(filepath, encoding='utf-8-sig'):
    with open(filepath, 'r', encoding=encoding) as file_obj:
        data = []
        reader = csv.reader(file_obj)
        for row in reader:
            data.append(row)

        return data
def data_prepocessing(data):
    current_skin_T = []
    current_CPU_0_usage = []
    current_CPU_1_usage = []
    current_CPU_2_usage = []
    current_CPU_3_usage = []
    current_CPU_4_usage = []
    current_CPU_5_usage = []
    current_CPU_6_usage = []
    current_CPU_7_usage = []
    current_CPU_0 = []
    current_CPU_1 = []
    current_CPU_2 = []
    current_CPU_3 = []
    current_CPU_4 = []
    current_CPU_5 = []
    current_CPU_6 = []
    current_CPU_7 = []
    current_CPU_8 = []
    current_CPU_9 = []
    current_GPU_0 = []
    current_GPU_1 = []
    battert_viltage = []
    battert_current = []
    frequency_CPU_0 = []
    frequency_CPU_1 = []
    frequency_CPU_2 = []
    frequency_CPU_3 = []
    frequency_CPU_4 = []
    frequency_CPU_5 = []
    frequency_CPU_6 = []
    frequency_CPU_7 = []
    Total_utilization = []
    time_stamp = []
    downlink = []
    uplink = []
    
    CPU_EDC_through_time = []
    CPU_EDC_weight_through_time = []
    cluster_EDC_through_time = []
    cluster_EDC_weight_through_time = []
    for key, value in data.items():
        time_stamp.append(int(key))
        current_skin_T.append(float(value[0]))
        Total_utilization.append(float(value[1]))
        current_CPU_0_usage.append(float(value[2]))
        current_CPU_1_usage.append(float(value[3]))
        current_CPU_2_usage.append(float(value[4]))
        current_CPU_3_usage.append(float(value[5]))
        current_CPU_4_usage.append(float(value[6]))
        current_CPU_5_usage.append(float(value[7]))
        current_CPU_6_usage.append(float(value[8]))
        current_CPU_7_usage.append(float(value[9]))
        current_CPU_0.append(float(value[10]))
        current_CPU_1.append(float(value[11]))
        current_CPU_2.append(float(value[12]))
        current_CPU_3.append(float(value[13]))
        current_CPU_4.append(float(value[14]))
        current_CPU_5.append(float(value[15]))
        current_CPU_6.append(float(value[16]))
        current_CPU_7.append(float(value[17]))
        current_CPU_8.append(float(value[18]))
        current_CPU_9.append(float(value[19]))
        current_GPU_0.append(float(value[20]))
        current_GPU_1.append(float(value[21]))
        battert_viltage.append(float(value[22]))
        battert_current.append(float(value[23]))
        frequency_CPU_0.append(float(value[24]))
        frequency_CPU_1.append(float(value[25]))
        frequency_CPU_2.append(float(value[26]))
        frequency_CPU_3.append(float(value[27]))
        frequency_CPU_4.append(float(value[28]))
        frequency_CPU_5.append(float(value[29]))
        frequency_CPU_6.append(float(value[30]))
        frequency_CPU_7.append(float(value[31]))
        downlink.append(float(value[-2]))
        uplink.append(float(value[-1]))
        
        CPU_state_entries = {}
        CPU_state_time = {}
        EDC= {}
        EDC_weight = {}
        CPU_cluster_entries = {}
        CPU_cluster_time = {}
        EDC_cluster = {}
        EDC_cluster_weight = {}
        CPU_cluster_entries['CPU_0t5_1'] = 0
        CPU_cluster_entries['CPU_0t5_2'] = 0
        CPU_cluster_entries['CPU_0t5_3'] = 0
        CPU_cluster_time['CPU_0t5_1'] = 0
        CPU_cluster_time['CPU_0t5_2'] = 0
        CPU_cluster_time['CPU_0t5_3'] = 0
        CPU_cluster_time['CPU_0t5_idle'] = 0
        
        for i in range(8):
            CPU_state_entries['CPU_' + str(i) + '_1'] = float(value[32 + 3*i])
            CPU_state_entries['CPU_' + str(i) + '_2'] = float(value[33 + 3*i])
            CPU_state_entries['CPU_' + str(i) + '_3'] = float(value[34 + 3*i])
            if i < 6:
                   CPU_cluster_entries['CPU_0t5_1'] += CPU_state_entries['CPU_' + str(i) + '_1']
                   CPU_cluster_entries['CPU_0t5_2'] += CPU_state_entries['CPU_' + str(i) + '_2']
                   CPU_cluster_entries['CPU_0t5_3'] += CPU_state_entries['CPU_' + str(i) + '_3']
            else:
                CPU_cluster_entries['CPU_'+ str(i) +'_1'] = CPU_state_entries['CPU_' + str(i) + '_1']
                CPU_cluster_entries['CPU_'+ str(i) +'_2'] = CPU_state_entries['CPU_' + str(i) + '_2']
                CPU_cluster_entries['CPU_'+ str(i) +'_3'] = CPU_state_entries['CPU_' + str(i) + '_3']
        current_CPU_U=[]
        for i in range(8):
            CPU_state_time['CPU_' + str(i) + '_1'] = float(value[32+24 + 3*i])
            CPU_state_time['CPU_' + str(i) + '_2'] = float(value[33+24 + 3*i])
            CPU_state_time['CPU_' + str(i) + '_3'] = float(value[34+24 + 3*i])
            CPU_state_time['CPU_' + str(i) + '_idle'] = float(value[32+24 + 3*i]) + float(value[33+24 + 3*i]) + float(value[34+24 + 3*i])
            current_CPU_U.append(1-CPU_state_time['CPU_' + str(i) + '_idle']/1e6)
            
            if i < 6:
                   CPU_cluster_time['CPU_0t5_1'] += CPU_state_time['CPU_' + str(i) + '_1']
                   CPU_cluster_time['CPU_0t5_2'] += CPU_state_time['CPU_' + str(i) + '_2']
                   CPU_cluster_time['CPU_0t5_3'] += CPU_state_time['CPU_' + str(i) + '_3']
                   CPU_cluster_time['CPU_0t5_idle'] += CPU_cluster_time['CPU_0t5_1'] + CPU_cluster_time['CPU_0t5_2'] + CPU_cluster_time['CPU_0t5_3']
            else:
                CPU_cluster_time['CPU_'+ str(i) +'_1'] = CPU_state_time['CPU_' + str(i) + '_1']
                CPU_cluster_time['CPU_'+ str(i) +'_2'] = CPU_state_time['CPU_' + str(i) + '_2']
                CPU_cluster_time['CPU_'+ str(i) +'_3'] = CPU_state_time['CPU_' + str(i) + '_3']
                CPU_cluster_time['CPU_' + str(i) + '_idle'] = CPU_state_time['CPU_' + str(i) + '_idle']

        for i in range(8):
            if CPU_state_entries['CPU_' + str(i) + '_1'] == 0:
                EDC['CPU_'+ str(i) + '_1'] = 0
            else:
                EDC['CPU_'+ str(i) + '_1'] = CPU_state_time['CPU_' + str(i) + '_1']/CPU_state_entries['CPU_' + str(i) + '_1']
            if CPU_state_entries['CPU_' + str(i) + '_2'] == 0:
                EDC['CPU_'+ str(i) + '_2'] = 0
            else:
                 EDC['CPU_'+ str(i) + '_2'] = CPU_state_time['CPU_' + str(i) + '_2']/CPU_state_entries['CPU_' + str(i) + '_2']
            if CPU_state_entries['CPU_' + str(i) + '_3'] == 0:
                EDC['CPU_'+ str(i) + '_3'] = 0
            else:
                EDC['CPU_'+ str(i) + '_3'] = CPU_state_time['CPU_' + str(i) + '_3']/CPU_state_entries['CPU_' + str(i) + '_3']

            if i < 6:
                if CPU_cluster_entries['CPU_0t5_1'] == 0:
                   EDC_cluster['CPU_0t5_1'] = 0
                else:
                    EDC_cluster['CPU_0t5_1'] = CPU_cluster_time['CPU_0t5_1']/CPU_cluster_entries['CPU_0t5_1']
                if CPU_cluster_entries['CPU_0t5_2'] == 0:
                   EDC_cluster['CPU_0t5_2'] = 0
                else:
                    EDC_cluster['CPU_0t5_2'] = CPU_cluster_time['CPU_0t5_2']/CPU_cluster_entries['CPU_0t5_2']
                if CPU_cluster_entries['CPU_0t5_3'] == 0:
                    EDC_cluster['CPU_0t5_3'] = 0
                else:
                    EDC_cluster['CPU_0t5_3'] = CPU_cluster_time['CPU_0t5_3']/CPU_cluster_entries['CPU_0t5_3']
            else:
                if CPU_cluster_entries['CPU_' + str(i) + '_1'] == 0:
                    EDC_cluster['CPU_'+ str(i) + '_1'] = 0
                else:
                    EDC_cluster['CPU_'+ str(i) +'_1'] = CPU_cluster_time['CPU_'+ str(i) +'_1']/CPU_cluster_entries['CPU_'+ str(i) +'_1']
                if CPU_cluster_entries['CPU_' + str(i) + '_2'] == 0:
                    EDC_cluster['CPU_'+ str(i) + '_2'] = 0
                else:
                    EDC_cluster['CPU_'+ str(i) +'_2'] = CPU_cluster_time['CPU_'+ str(i) +'_2']/CPU_cluster_entries['CPU_'+ str(i) +'_2']
                if CPU_cluster_entries['CPU_' + str(i) + '_3'] == 0:
                    EDC_cluster['CPU_'+ str(i) + '_3'] = 0
                else:
                    EDC_cluster['CPU_'+ str(i) +'_3'] = CPU_cluster_time['CPU_'+ str(i) +'_3']/CPU_cluster_entries['CPU_'+ str(i) +'_3']

            if CPU_state_time['CPU_' + str(i) + '_idle'] ==0:
               EDC_weight['CPU_'+ str(i) + '_1'] = 0
               EDC_weight['CPU_'+ str(i) + '_2'] = 0
               EDC_weight['CPU_'+ str(i) + '_3'] = 0
            else:
                EDC_weight['CPU_'+ str(i) + '_1'] = CPU_state_time['CPU_' + str(i) + '_1']/CPU_state_time['CPU_' + str(i) + '_idle']
                EDC_weight['CPU_'+ str(i) + '_2'] = CPU_state_time['CPU_' + str(i) + '_2']/CPU_state_time['CPU_' + str(i) + '_idle']
                EDC_weight['CPU_'+ str(i) + '_3'] = CPU_state_time['CPU_' + str(i) + '_3']/CPU_state_time['CPU_' + str(i) + '_idle']
            if i < 6:
                if CPU_cluster_time['CPU_0t5_idle'] ==0:
                    EDC_cluster_weight['CPU_0t5_1'] = 0
                    EDC_cluster_weight['CPU_0t5_2'] = 0
                    EDC_cluster_weight['CPU_0t5_3'] = 0
                else:
                    EDC_cluster_weight['CPU_0t5_1'] = CPU_cluster_time['CPU_0t5_1']/CPU_cluster_time['CPU_0t5_idle']
                    EDC_cluster_weight['CPU_0t5_2'] = CPU_cluster_time['CPU_0t5_2']/CPU_cluster_time['CPU_0t5_idle']
                    EDC_cluster_weight['CPU_0t5_3'] = CPU_cluster_time['CPU_0t5_3']/CPU_cluster_time['CPU_0t5_idle']
            else:
                 EDC_cluster_weight['CPU_'+ str(i) +'_1'] = EDC_weight['CPU_'+ str(i) + '_1']
                 EDC_cluster_weight['CPU_'+ str(i) +'_2'] = EDC_weight['CPU_'+ str(i) + '_2']
                 EDC_cluster_weight['CPU_'+ str(i) +'_3'] = EDC_weight['CPU_'+ str(i) + '_3']

        CPU_EDC_through_time.append(EDC)
        CPU_EDC_weight_through_time.append(EDC_weight)
        cluster_EDC_through_time.append(EDC_cluster)
        cluster_EDC_weight_through_time.append(EDC_cluster_weight)

    # data prepocessing
    Current_skin_T = np.array(current_skin_T)
    voltage = np.array(battert_viltage)
    current = -np.array(battert_current)
    frequency_CPU_0 = np.array(frequency_CPU_0)
    frequency_CPU_1 = np.array(frequency_CPU_1)
    frequency_CPU_2 = np.array(frequency_CPU_2)
    frequency_CPU_3 = np.array(frequency_CPU_3)
    frequency_CPU_4 = np.array(frequency_CPU_4)
    frequency_CPU_5 = np.array(frequency_CPU_5)
    frequency_CPU_6 = np.array(frequency_CPU_6)
    frequency_CPU_7 = np.array(frequency_CPU_7)

    current_CPU_0_T = np.array(current_CPU_0)
    current_CPU_1_T = np.array(current_CPU_1)
    current_CPU_2_T = np.array(current_CPU_2)
    current_CPU_3_T = np.array(current_CPU_3)
    current_CPU_4_T = np.array(current_CPU_4)
    current_CPU_5_T = np.array(current_CPU_5)
    current_CPU_6_T = np.array(current_CPU_6)
    current_CPU_7_T = np.array(current_CPU_7)
    current_CPU_8_T = np.array(current_CPU_8)
    current_CPU_9_T = np.array(current_CPU_8)

    average_CPU_T = (current_CPU_0_T + current_CPU_1_T + current_CPU_2_T + current_CPU_3_T + \
                        current_CPU_4_T + current_CPU_5_T + current_CPU_6_T + current_CPU_7_T)/8

    CPU_0_5_average_U = (np.array(current_CPU_0_usage) + np.array(current_CPU_1_usage) + np.array(current_CPU_2_usage) +\
            np.array(current_CPU_3_usage) + np.array(current_CPU_4_usage) + np.array(current_CPU_5_usage))/6
    CPU_6_average_U = np.array(current_CPU_6_usage)
    CPU_7_average_U = np.array(current_CPU_7_usage)
    CPU_0_average_U = np.array(current_CPU_0_usage)
    CPU_1_average_U = np.array(current_CPU_1_usage)
    CPU_2_average_U = np.array(current_CPU_2_usage)
    CPU_3_average_U = np.array(current_CPU_3_usage)
    CPU_4_average_U = np.array(current_CPU_4_usage)
    CPU_5_average_U = np.array(current_CPU_5_usage)
    CPU_0_2_5 = (CPU_0_average_U, CPU_1_average_U, CPU_2_average_U, CPU_3_average_U, CPU_4_average_U, CPU_5_average_U)

    power = voltage * current / 1e9
    average_frequency = (frequency_CPU_0 + frequency_CPU_1 + frequency_CPU_2 + frequency_CPU_3 + \
                        frequency_CPU_5 + frequency_CPU_6 + frequency_CPU_7)/8
    Total_utilization = np.array(Total_utilization)

    WEDC = np.zeros((len(CPU_EDC_through_time), 24))
    for t in range(len(CPU_EDC_through_time)):
         for i in range(8):
             WEDC[t,3*i+0] = CPU_EDC_through_time[t]['CPU_' + str(i) + '_1'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i) + '_1']
             WEDC[t,3*i+1] = CPU_EDC_through_time[t]['CPU_' + str(i) + '_2'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i) + '_2']
             WEDC[t,3*i+2] = CPU_EDC_through_time[t]['CPU_' + str(i) + '_3'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i) + '_3']

    WEDC_cluster = np.zeros((len(CPU_EDC_through_time), 9))
    for t in range(len(CPU_EDC_through_time)):
        for i in range(3):
           if i <3:
                WEDC_cluster[t,3*i+0] = cluster_EDC_through_time[t]['CPU_0t5_1'] * cluster_EDC_weight_through_time[t]['CPU_0t5_1']
                WEDC_cluster[t,3*i+1] = cluster_EDC_through_time[t]['CPU_0t5_2'] * cluster_EDC_weight_through_time[t]['CPU_0t5_2']
                WEDC_cluster[t,3*i+2] = cluster_EDC_through_time[t]['CPU_0t5_3'] * cluster_EDC_weight_through_time[t]['CPU_0t5_3']
           else:
               WEDC_cluster[t,3*i+0] = cluster_EDC_through_time[t]['CPU_' + str(i+5) + '_1'] * cluster_EDC_weight_through_time[t]['CPU_' + str(i+5) + '_1']
               WEDC_cluster[t,3*i+1] = cluster_EDC_through_time[t]['CPU_' + str(i+5) + '_2'] * cluster_EDC_weight_through_time[t]['CPU_' + str(i+5) + '_2']
               WEDC_cluster[t,3*i+2] = cluster_EDC_through_time[t]['CPU_' + str(i+5) + '_3'] * cluster_EDC_weight_through_time[t]['CPU_' + str(i+5) + '_3']

    # WEDC = np.concatenate((WEDC[:, 2].reshape(-1,1),WEDC[:, 5].reshape(-1,1),WEDC[:, 8].reshape(-1,1),WEDC[:, 11].reshape(-1,1),WEDC[:, 14].reshape(-1,1),WEDC[:, 17].reshape(-1,1)),1)
    # WEDC  = WEDC[:,9:18].reshape(WEDC.shape[0],-1)

    downlink = np.array(downlink)
    uplink = np.array(uplink)
    return average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5, frequency_CPU_6, \
           frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5,WEDC, time_stamp,\
           downlink, uplink, Current_skin_T


# prepare training data
original_data = read_json('./json_data.json')
test_data = read_json('./json_data_test.json')

average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5,frequency_CPU_6,frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5, WEDC,time_stamp, downlink, uplink,Current_skin_T = data_prepocessing(original_data)
N_train = Total_utilization.shape[0]
CPU_0_average_U, CPU_1_average_U, CPU_2_average_U,CPU_3_average_U,CPU_4_average_U,CPU_5_average_U = CPU_0_2_5
time_stamp = np.array(time_stamp)
time_stamp = time_stamp.astype(int)
time_stamp = (time_stamp - time_stamp[0])/1000

time_stamp_start = 1657301923627 # cacluate channel number
time_stamp_nsg_start = int(1657301936 * 1000)
time_stamp_nsg_stop = int(1657302839 *1000)
deviation =  time_stamp_nsg_start-time_stamp_start
print(deviation)

time_stamp_point1 = 1657302120 *1000 #4nrc-1nrc
time_stamp_point2 = 1657302350 *1000 #1nrc-4nrc
time_stamp_point3 = 1657302416 *1000 #4nrc-1nrc

# time_stamp_nsg = np.arange(time_stamp_nsg_start,time_stamp_nsg_stop)
time_stamp_nsg = np.linspace(time_stamp_nsg_start,time_stamp_nsg_stop, num=time_stamp.shape[0])
point1 = np.where(np.abs(time_stamp_nsg - time_stamp_point1) == np.amin(np.abs(time_stamp_nsg - time_stamp_point1)))
point2 = np.where(np.abs(time_stamp_nsg - time_stamp_point2) == np.amin(np.abs(time_stamp_nsg - time_stamp_point2)))
point3 = np.where(np.abs(time_stamp_nsg - time_stamp_point3) == np.amin(np.abs(time_stamp_nsg - time_stamp_point3)))

channe_number = np.ones((time_stamp_nsg.shape[0],))*4
# channe_number[0:point1[0].item()] = channe_number[0,point1[0].item()]
channe_number[point1[0].item():point2[0].item()] = channe_number[point1[0].item():point2[0].item()]/4
# channe_number[point2[0].item():point3[0].item()] = channe_number[point2[0].item():point3[0].item()]/4
channe_number[point3[0].item():] = channe_number[point3[0].item():]/4


# power_original = np.array(power)
# pwer = pd.Series(power)
# pwer = pwer.rolling(window=5).mean()
# power[5:] = pwer[5:]


SETTING_CONDITION = 5 # 0,1,2,3,4

if SETTING_CONDITION == 0:
    X_train = np.array([Total_utilization.reshape(N_train,-1),channe_number.reshape(N_train,1),
                        np.ones((N_train,1))])
elif SETTING_CONDITION == 1:
    X_train = np.array([Total_utilization.reshape(N_train,-1),
                        average_CPU_T.reshape(N_train,-1),channe_number.reshape(N_train,1),np.ones((N_train,1))])
elif SETTING_CONDITION == 2:
    X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_average_U.reshape(N_train,-1), CPU_1_average_U.reshape(N_train,-1), CPU_2_average_U.reshape(N_train,-1), CPU_3_average_U.reshape(N_train,-1),
                        CPU_4_average_U.reshape(N_train,-1), CPU_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),channe_number.reshape(N_train,1),
                        np.ones((N_train,1))])
    X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
                        np.ones((N_train,1))])

elif SETTING_CONDITION == 3 or SETTING_CONDITION == 4:
    X_train = np.array([Total_utilization.reshape(N_train,-1),
                        CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1), average_CPU_T.reshape(N_train,-1),channe_number.reshape(N_train,1),np.ones((N_train,1))])
    X_train = np.array([Total_utilization.reshape(N_train,-1),
                        CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1), average_CPU_T.reshape(N_train,-1),np.ones((N_train,1))])
elif SETTING_CONDITION == 5:
    X_train = np.array([Total_utilization.reshape(N_train,-1),
                        CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1), average_CPU_T.reshape(N_train,-1), downlink.reshape(N_train,-1)])

X_train = X_train.transpose(0,2,1).reshape((X_train.shape[0], N_train))
Train_X = X_train.transpose()
train_y = power.reshape(power.shape[0], 1)
if SETTING_CONDITION == 4:
    Train_X = np.concatenate((WEDC, Train_X),1)
M_train = Train_X.shape[1]


# prepare test data
average_frequency_test, Total_utilization_test, power_test, average_CPU_T_test, frequency_CPU_5_test,frequency_CPU_6_test,frequency_CPU_7_test, CPU_0_5_average_U_test, CPU_6_average_U_test, CPU_7_average_U_test, CPU_0_2_5_test, WEDC_test, time_stamp_test, downlink_test, uplink_test,Current_skin_T_test= data_prepocessing(test_data)
N_test = Total_utilization_test.shape[0]
CPU_0_average_U_test, CPU_1_average_U_test, CPU_2_average_U_test,CPU_3_average_U_test,CPU_4_average_U_test,CPU_5_average_U_test = CPU_0_2_5_test
time_stamp_test = np.array(time_stamp_test)
time_stamp_test =  time_stamp_test.astype(int)
time_stamp_test = (time_stamp_test -  time_stamp_test[0])/1000

time_stamp_start_test = 1657303156433
time_stamp_nsg_start_test = int(1657303169 * 1000)
time_stamp_nsg_stop_test = int(1657304072 *1000)

deviation =  time_stamp_nsg_start_test-time_stamp_start_test
print(deviation)
time_stamp_point1_test = 1657303277 *1000 #4nrc-1nrc

time_stamp_point1_test = int(time_stamp_point1)

time_stamp_nsg_test = np.arange(time_stamp_nsg_start_test,time_stamp_nsg_stop_test)
time_stamp_nsg_test = np.linspace(time_stamp_nsg_start_test,time_stamp_nsg_stop_test, num=time_stamp_test.shape[0])
point1_test = np.where(np.abs(time_stamp_nsg_test - time_stamp_point1_test) == np.amin(np.abs(time_stamp_nsg_test - time_stamp_point1_test)))

channe_number_test = np.ones((time_stamp_nsg_test.shape[0],))*4
channe_number_test[point1_test[0].item():] = channe_number_test[point1_test[0].item():]/4


if SETTING_CONDITION == 3:
    X_test = np.array([Total_utilization_test.reshape(N_test,-1),channe_number_test.reshape(N_test,-1),
                        np.ones((N_test,1))])
elif SETTING_CONDITION == 1:
    X_test = np.array([Total_utilization_test.reshape(N_test,-1),
                        average_CPU_T_test.reshape(N_test,-1),channe_number_test.reshape(N_test,-1),np.ones((N_test,1))])
elif SETTING_CONDITION == 2:
    X_test = np.array([Total_utilization_test.reshape(N_test,-1),CPU_0_average_U_test.reshape(N_test,-1), CPU_1_average_U_test.reshape(N_test,-1), CPU_2_average_U_test.reshape(N_test,-1),CPU_3_average_U_test.reshape(N_test,-1),
                       CPU_4_average_U_test.reshape(N_test,-1), CPU_5_average_U_test.reshape(N_test,-1), CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1), channe_number_test.reshape(N_test,-1),
                       np.ones((N_test,1))])
    X_test = np.array([Total_utilization_test.reshape(N_test,-1), CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1),
                       np.ones((N_test,1))])
elif SETTING_CONDITION == 3 or SETTING_CONDITION == 4:
    X_test = np.array([Total_utilization_test.reshape(N_test,-1), 
                        CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1), average_CPU_T_test.reshape(N_test,-1),np.ones((N_test,1))])
elif SETTING_CONDITION == 5:
     X_test = np.array([Total_utilization_test.reshape(N_test,-1),
                        CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1), average_CPU_T_test.reshape(N_test,-1),downlink_test.reshape(N_test,-1)])

X_test = X_test.transpose(0,2,1).reshape((X_test.shape[0], N_test))
Test_X = X_test.transpose()
test_y = power_test.reshape(power_test.shape[0], 1)
if SETTING_CONDITION == 4:
   Test_X = np.concatenate((WEDC_test, Test_X),1)
M_test = Test_X.shape[1]

# Ordinary Linear Regression
base_power_iperf =750
base_power_5g = 3900 - base_power_iperf

# train_y = train_y-base_power_iperf
# train_y[train_y<0] = 0
# w=np.zeros((M_train,1))
# w=np.linalg.pinv(Train_X).dot(train_y) # pinv(A)*b
# Predict_y_OLS=Train_X.dot(w)
# np.savetxt('Original.txt', w)
# Predict_y_OLS = Predict_y_OLS+base_power_iperf
# train_y = train_y+base_power_iperf
train_yy = np.array(train_y)

# train_yy = train_y

for i in range(N_train):
    if downlink[i]>10:
        train_y[i] = train_y[i]-base_power_iperf - base_power_5g
    else:
        train_y[i] = train_y[i]-100000
        print(train_y[i])

train_y[train_y<0] = 0
delete_points = np.where(train_y==0)[0]


w=np.zeros((M_train,1))
w=np.linalg.pinv(Train_X).dot(train_y) # pinv(A)*b
Predict_y_OLS=Train_X.dot(w)
np.savetxt('Original.txt', w)

for i in range(N_train):
    if downlink[i]>10:
        Predict_y_OLS[i] = Predict_y_OLS[i]+base_power_iperf+base_power_5g
        train_y[i] = train_y[i]+base_power_iperf + base_power_5g
        
    else:
        Predict_y_OLS[i] = Predict_y_OLS[i]+base_power_iperf
        train_y[i] = train_y[i]+base_power_iperf

# Predict_y_OLS = Predict_y_OLS+base_power_iperf
# train_y = train_y+base_power_iperf




# plot the curve of OLS
plt.figure(1)
plt.plot(time_stamp,train_yy, 'go-', linewidth=1.5)
plt.plot(time_stamp,Predict_y_OLS, 'r*-', linewidth=1.5)
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("Ordinary Lineary Regression Training Results")
plt.legend(['Raw Data', 'Regreassion'])
plt.savefig('Training results.png')

#validation

Predict_y_OLS_test=Test_X.dot(w)
for i in range(N_test):
    if downlink_test[i]>10:
        Predict_y_OLS_test[i] = Predict_y_OLS_test[i]+base_power_iperf+base_power_5g
    else:
        Predict_y_OLS_test[i] = Predict_y_OLS_test[i]+base_power_iperf


plt.figure(2)
plt.plot(time_stamp_test, test_y, 'go-', linewidth=1.5)
plt.plot(time_stamp_test, Predict_y_OLS_test, 'r*-', linewidth=1.5)
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("Ordinary Lineary Regression Validation Results")
plt.legend(['Raw Data', 'Regreassion'])
plt.savefig('Testing results.png')


EMS_tr=np.sum((train_y-Predict_y_OLS)**2)/2
EMS_te=np.sum((test_y-Predict_y_OLS_test)**2)/2
# print(w)
ERMS_tr=np.sqrt(2*EMS_tr/N_train)
ERMS_te=np.sqrt(2*EMS_te/N_test)

correlation_train = np.mean((Predict_y_OLS-np.mean(Predict_y_OLS))*(train_y-np.mean(train_y)))/(np.mean((Predict_y_OLS-np.mean(Predict_y_OLS))**2) * np.mean((train_y-np.mean(train_y))**2))**0.5
correlation_test = np.mean((Predict_y_OLS_test-np.mean(Predict_y_OLS_test))*(test_y-np.mean(test_y)))/(np.mean((Predict_y_OLS_test-np.mean(Predict_y_OLS_test))**2) * np.mean((test_y-np.mean(test_y))**2))**0.5

print(f'OLS_correlation_train = {correlation_train}\nOLS_correlation_test = {correlation_test}\n')

print(f'OLS_ERMS_tr_= {ERMS_tr}\nOLS_ERMS_te = {ERMS_te}\n')

MS_tr=np.sum(1-np.abs(train_yy-Predict_y_OLS)/train_yy)
MS_te=np.sum(1-np.abs(test_y-Predict_y_OLS_test)/test_y)
average_accuracy_tr = MS_tr/N_train
average_accuracy_te = MS_te/N_test
print(f'average_accuracy_tr_= {average_accuracy_tr}\naverage_accuracy_te = {average_accuracy_te}\n')

# locally weighted Linear Regression

# Weight Matrix in code. It is a diagonal matrix.
def wm(x_point, Train_X, tau):

    n = Train_X.shape[0]
    # Initialising R as an identity matrix.
    R = np.eye(n)
    # Calculating weights for all training examples [x(i)'s].
    for i in range(n):
        xi = Train_X[i,:]
        d = (-2 * tau * tau)
        R[i, i] = np.exp(np.dot((xi-x_point), (xi-x_point).T)/d)

    return R

def predict(Train_X, y, x_point, tau):

    # calculate the weight
    R = wm(x_point, Train_X, tau)
    # Calculating parameter theta using the formula.
    w = np.linalg.pinv(Train_X.T.dot(R.dot(Train_X))).dot(Train_X.T.dot(R.dot(y)))
    # Calculating predictions.
    pred = np.dot(x_point, w)
    # Returning the theta and prediction
    return w, pred


predict_y_Local = []
Theta = []
tau=0.8
# Predicting for all nval values and storing them in preds.
for point in range(N_train):
    x_point=Train_X[point,:].reshape((1,M_train))
    theta, pred = predict(Train_X, train_y, x_point, tau)
    predict_y_Local.append(pred)
    Theta.append(theta)
# change prediction type and reshaping
predict_y_Local=np.array(predict_y_Local)
predict_y_Local=predict_y_Local.reshape((predict_y_Local.shape[0],-1))


plt.figure(3)
plt.plot(time_stamp,train_y, 'go-', linewidth=1.5)
plt.plot(time_stamp,predict_y_Local, 'r*-', linewidth=1.5)
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("Local weighted Lineary Regression Training Results")
plt.legend(['Raw Data', 'Regreassion'])

#validation
predict_y_Local_test = []
tau= 0.8
# Predicting for all nval values and storing them in preds.
for point in range(N_test):
    x_point=Test_X[point,:].reshape((1,M_test))
    w, pred = predict(Train_X, train_y, x_point, tau)
    predict_y_Local_test.append(pred)
# change prediction type and reshaping
predict_y_Local_test=np.array(predict_y_Local_test)
predict_y_Local_test=predict_y_Local_test.reshape((predict_y_Local_test.shape[0],-1))

EMS_tr=np.sum((train_y-predict_y_Local)**2)/2
EMS_te=np.sum((test_y-predict_y_Local_test)**2)/2
# print(w)
ERMS_tr=np.sqrt(2*EMS_tr/N_train)
ERMS_te=np.sqrt(2*EMS_te/N_test)

correlation_train = np.mean((predict_y_Local-np.mean(predict_y_Local))*(train_y-np.mean(train_y)))/(np.mean((predict_y_Local-np.mean(predict_y_Local))**2) * np.mean((train_y-np.mean(train_y))**2))**0.5
correlation_test = np.mean((predict_y_Local_test-np.mean(predict_y_Local_test))*(test_y-np.mean(test_y)))/(np.mean((predict_y_Local_test-np.mean(predict_y_Local_test))**2) * np.mean((test_y-np.mean(test_y))**2))**0.5
# correlation_test = np.corrcoef(predict_y_Local, test_y)

print(f'LWLR_correlation_train= {correlation_train}\nLWLR_correlation_test = {correlation_test}\n')
print(f'LWLR_ERMS_tr_= {ERMS_tr}\nLWLR_ERMS_te = {ERMS_te}\n')

plt.figure(4)
plt.plot(time_stamp_test,test_y, 'go-', linewidth=1.5)
plt.plot(time_stamp_test,predict_y_Local_test, 'r*-', linewidth=1.5)
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("Local weighted Lineary Regression Training Results")
plt.legend(['Raw Data', 'Regreassion'])


# Non-Linear Rergression using Temperature

X_train = np.concatenate((average_CPU_T.reshape(1, N_train), time_stamp.reshape(1, N_train)),0)
y_train = power.reshape(-1,)
X_test = np.concatenate((average_CPU_T_test.reshape(1, N_test),  time_stamp_test.reshape(1, N_test)),0)
y_test = power_test.reshape(-1, )
# # Non-Linear Regression
# def func(X, R_E, C, A):
#     V0 = 23
#     return (X[0,:] - V0 - A * np.exp(-X[1,:]/(R_E*C))) /  R_E

# popt, pcov = curve_fit(func, X_train, y_train)
# print(popt)

# #training
# plt.figure(5)
# plt.plot(time_stamp, y_train, 'go-', linewidth=1)
# plt.plot(time_stamp, func(X_train, *popt), 'r*-', linewidth=1, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.xlabel("training")
# plt.ylabel("Prediction")
# plt.title("None Lineary Regression")
# plt.legend(['Raw Data', 'Regreassion'])
# predict_NL = func(X_train, *popt)

# #training
# plt.figure(6)
# plt.plot(time_stamp_test, y_test, 'go-', linewidth=1)
# plt.plot(time_stamp_test, func(X_test, *popt), 'r*-', linewidth=1, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
# plt.xlabel("testing")
# plt.ylabel("Prediction")
# plt.title("None Lineary Regression")
# plt.legend(['Raw Data', 'Regreassion'])
# predict_NL_test = func(X_test, *popt)

# EMS_tr=np.sum((y_train-predict_NL)**2)/2
# EMS_te=np.sum((y_test-predict_NL_test)**2)/2
# ERMS_tr=np.sqrt(2*EMS_tr/N_train)
# ERMS_te=np.sqrt(2*EMS_te/N_test)

# correlation_train = np.mean((predict_NL-np.mean(predict_NL))*(y_train-np.mean(y_train)))/(np.mean((predict_NL-np.mean(predict_NL))**2) * np.mean((y_train-np.mean(y_train))**2))**0.5
# correlation_test = np.mean((predict_NL_test-np.mean(predict_NL_test))*(y_test-np.mean(y_test)))/(np.mean((predict_NL_test-np.mean(predict_NL_test))**2) * np.mean((y_test-np.mean(y_test))**2))**0.5

# correlation_test = np.corrcoef(predict_NL_test, y_test)



# print(f'NL_correlation_train = {correlation_train}\nNL_correlation_test = {correlation_test}\n')
# print(f'NL_ERMS_tr_= {ERMS_tr}\nNL_ERMS_te = {ERMS_te}\n')

plt.figure(7)
plt.plot(time_stamp,train_y, 'go-', linewidth=1)
plt.plot(time_stamp,Predict_y_OLS, 'r*-', linewidth=1)
# plt.plot(time_stamp,predict_NL, 'y-*', linewidth=1) # Predictions in blue color.
plt.title("OLR and LWLR Training Results Comparison")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'LR', 'NLR'])

plt.figure(8)
plt.plot(time_stamp_test,test_y, 'go-', linewidth=1)
plt.plot(time_stamp_test,Predict_y_OLS_test, 'r*-', linewidth=1)
# plt.plot(time_stamp_test,predict_NL_test, 'y-*', linewidth=1) # Predictions in blue color.
plt.title("Linear Regression and Non_Linear Validation Results Comparison")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'LR', 'NLR'])


plt.figure(9)
plt.plot(time_stamp,train_y, 'go-', linewidth=1)
plt.plot(time_stamp,Predict_y_OLS, 'r*-', linewidth=1)
plt.plot(time_stamp,predict_y_Local, 'b-*', linewidth=1) # Predictions in blue color.
plt.title("OLR and LWLR Training Results Comparison")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'OLR', 'LWLR'])

# CPU power
plt.figure(10)
plt.plot(time_stamp_test,test_y, 'go-', linewidth=1)
plt.plot(time_stamp_test,Predict_y_OLS_test, 'r*-', linewidth=1)
plt.plot(time_stamp_test,predict_y_Local_test, 'b-*', linewidth=1) # Predictions in blue color.
plt.title("OLR and LWLR Validation Results Comparison")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'OLR', 'LWLR'])


# seperate CPU power from Transceiver
beta1 = 0
beta2 = 0
beta3 = 0

cpu_power = beta1*Total_utilization.reshape(N_train,-1)+beta2*CPU_6_average_U.reshape(N_train,-1) +beta3*CPU_7_average_U.reshape(N_train,-1)+2000

plt.figure(11)
plt.plot(time_stamp,cpu_power, 'go-', linewidth=1)
plt.ylabel("Prediction")
plt.legend(['power_cpu_estimate'])

# CPU power
plt.figure(10)
plt.plot(time_stamp_test,test_y, 'go-', linewidth=1)
plt.plot(time_stamp_test,Predict_y_OLS_test, 'r*-', linewidth=1)
plt.plot(time_stamp_test,predict_y_Local_test, 'b-*', linewidth=1) # Predictions in blue color.
plt.title("OLR and LWLR Validation Results Comparison")
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'OLR', 'LWLR'])

beta1 = 4.893752682294748411e+02
beta2 = 0
beta3 = 0

cpu_power = beta1*Total_utilization.reshape(N_train,-1)+beta2*CPU_6_average_U.reshape(N_train,-1) +beta3*CPU_7_average_U.reshape(N_train,-1)+2000

plt.figure(11)
plt.plot(time_stamp,cpu_power, 'go-', linewidth=1)
plt.ylabel("Prediction")
plt.legend(['power_cpu_estimate'])

plt.show()

