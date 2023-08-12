import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_json(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as file_obj:
        return json.load(file_obj)

original_data = read_json('./json_data.json')

test_data = read_json('./json_data_test.json')
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
    
    CPU_EDC_through_time = []
    CPU_EDC_weight_through_time = []
    
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
        
        CPU_state_entries = {}
        CPU_state_time = {}
        EDC= {}
        EDC_weight = {}
        for i in range(8):
            CPU_state_entries['CPU_' + str(i) + '_1'] = float(value[32 + i%3 + 3*(i//3)])
            CPU_state_entries['CPU_' + str(i) + '_2'] = float(value[33 + i%3 + 3*(i//3)])
            CPU_state_entries['CPU_' + str(i) + '_3'] = float(value[34 + i%3 + 3*(i//3)])


        for i in range(8):
            CPU_state_time['CPU_' + str(i) + '_1'] = float(value[32+24 + i%3 + 3*(i//3)])
            CPU_state_time['CPU_' + str(i) + '_2'] = float(value[33+24 + i%3 + 3*(i//3)])
            CPU_state_time['CPU_' + str(i) + '_3'] = float(value[34+24 + i%3 + 3*(i//3)])
            CPU_state_time['CPU_' + str(i) + '_idle'] = float(value[32+24 + i%3 + 3*(i//3)]) + float(value[32+24 + i%3 + 3*(i//3)]) + float(value[34+24 + i%3 + 3*(i//3)])

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
            
            if CPU_state_time['CPU_' + str(i) + '_idle'] ==0:
               EDC_weight['CPU_'+ str(i) + '_1'] = 0
               EDC_weight['CPU_'+ str(i) + '_2'] = 0
               EDC_weight['CPU_'+ str(i) + '_3'] = 0
            else:
                EDC_weight['CPU_'+ str(i) + '_1'] = CPU_state_time['CPU_' + str(i) + '_1']/CPU_state_time['CPU_' + str(i) + '_idle']
                EDC_weight['CPU_'+ str(i) + '_2'] = CPU_state_time['CPU_' + str(i) + '_2']/CPU_state_time['CPU_' + str(i) + '_idle']
                EDC_weight['CPU_'+ str(i) + '_3'] = CPU_state_time['CPU_' + str(i) + '_3']/CPU_state_time['CPU_' + str(i) + '_idle']
            
        CPU_EDC_through_time.append(EDC)
        CPU_EDC_weight_through_time.append(EDC_weight)

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
         for i in range(8*3):
             WEDC[t,3*int(i/3)+i%3] = CPU_EDC_through_time[t]['CPU_' + str(i//3) + '_1'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i//3) + '_1']
             WEDC[t,3*int(i/3)+i%3] = CPU_EDC_through_time[t]['CPU_' + str(i//3) + '_2'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i//3) + '_2']
             WEDC[t,3*int(i/3)+i%3] = CPU_EDC_through_time[t]['CPU_' + str(i//3) + '_3'] * CPU_EDC_weight_through_time[t]['CPU_' + str(i//3) + '_3']


    return average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5, frequency_CPU_6, \
           frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5,WEDC, time_stamp


# prepare training data
average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5,frequency_CPU_6,frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5, WEDC, time_stamp = data_prepocessing(original_data)
N_train = Total_utilization.shape[0]

CPU_0_average_U, CPU_1_average_U, CPU_1_average_U,CPU_1_average_U,CPU_1_average_U,CPU_1_average_U = CPU_0_2_5

X_train = np.array([Total_utilization.reshape(N_train,-1),
                    average_CPU_T.reshape(N_train,-1),np.ones((N_train,1))])

X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
                    average_CPU_T.reshape(N_train,-1),np.ones((N_train,1))])

# X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
#                     CPU_0_average_U.reshape(N_train,-1), CPU_1_average_U.reshape(N_train,-1), CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),np.ones((N_train,1))])

# X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
#                     np.ones((N_train,1))])

X_train = X_train.transpose(0,2,1).reshape((X_train.shape[0], N_train))
Train_X = X_train.transpose()
train_y = power.reshape(power.shape[0], 1)

Train_X = np.concatenate((WEDC, Train_X),1)
M_train = Train_X.shape[1]

# prepare test data
average_frequency_test, Total_utilization_test, power_test, average_CPU_T_test, frequency_CPU_5_test,frequency_CPU_6_test,frequency_CPU_7_test, CPU_0_5_average_U_test, CPU_6_average_U_test, CPU_7_average_U_test, CPU_0_2_5_test, WEDC_test, time_stamp_test = data_prepocessing(test_data)
N_test = Total_utilization_test.shape[0]

CPU_0_average_U_test, CPU_1_average_U_test, CPU_1_average_U_test,CPU_1_average_U_test,CPU_1_average_U_test,CPU_1_average_U_test = CPU_0_2_5_test


X_test = np.array([Total_utilization_test.reshape(N_test,-1),
                   average_CPU_T_test.reshape(N_test,-1),np.ones((N_test,1))])

X_test = np.array([Total_utilization_test.reshape(N_test,-1), CPU_0_5_average_U_test.reshape(N_test,-1), CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1),
                   average_CPU_T_test.reshape(N_test,-1),np.ones((N_test,1))])



X_test = X_test.transpose(0,2,1).reshape((X_test.shape[0], N_test))
Test_X = X_test.transpose()
test_y = power_test.reshape(power_test.shape[0], 1)

Test_X = np.concatenate((WEDC_test, Test_X),1)
M_test = Test_X.shape[1]

time_stamp = np.array(time_stamp)
time_stamp = time_stamp.astype(int)

time_stamp = (time_stamp - time_stamp[0])/1000


time_stamp_test = np.array(time_stamp_test)
time_stamp_test =  time_stamp_test.astype(int)
time_stamp_test = ( time_stamp_test -  time_stamp_test[0])/1000


X_train = np.concatenate((average_CPU_T.reshape(1, N_train), time_stamp.reshape(1, N_train)),0)
y_train = power.reshape(-1,)
X_test = np.concatenate((average_CPU_T_test.reshape(1, N_test),  time_stamp_test.reshape(1, N_test)),0)
y_test = power_test.reshape(-1, )
# Non-Linear Regression
print(X_train.shape)
print(y_train.shape)
def func(X, R_E,C, A):
    V0 = 23
    # C= float('inf')
    return (X[0,:] - V0 - A * np.exp(-X[1,:]/(R_E*C))) /  R_E

popt, pcov = curve_fit(func, X_train, y_train)
print(popt)

#training
plt.figure(1)
plt.plot(time_stamp, y_train, 'go-', linewidth=1.5)
plt.plot(time_stamp, func(X_train, *popt), 'r*-', linewidth=1.5, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("None Lineary Regression")
plt.legend(['Raw Data', 'Regreassion'])
predict_NL = func(X_train, *popt)

#training
plt.figure(2)
plt.plot(time_stamp_test, y_test, 'go-', linewidth=1.5)
plt.plot(time_stamp_test, func(X_test, *popt), 'r*-', linewidth=1.5, label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xlabel("Elapsed Time (s)")
plt.ylabel("Prediction (mW)")
plt.title("None Lineary Regression")
plt.legend(['Raw Data', 'Regreassion'])
predict_NL_test = func(X_test, *popt)

EMS_tr=np.sum((y_train-predict_NL)**2)/2
EMS_te=np.sum((y_test-predict_NL_test)**2)/2
# print(w)
ERMS_tr=np.sqrt(2*EMS_tr/N_train)
ERMS_te=np.sqrt(2*EMS_te/N_test)

correlation_train = np.mean((predict_NL-np.mean(predict_NL))*(y_train-np.mean(y_train)))/(np.mean((predict_NL-np.mean(predict_NL))**2) * np.mean((y_train-np.mean(y_train))**2))**0.5
correlation_test = np.mean((predict_NL_test-np.mean(predict_NL_test))*(y_test-np.mean(y_test)))/(np.mean((predict_NL_test-np.mean(predict_NL_test))**2) * np.mean((y_test-np.mean(y_test))**2))**0.5

print(f'NL_correlation_train = {correlation_train}\nNL_correlation_test = {correlation_test}\n')
print(f'NL_ERMS_tr_= {ERMS_tr}\nNL_ERMS_te = {ERMS_te}\n')
# plt.show()


MS_tr=np.sum(1-np.abs(y_train-predict_NL)/y_train)
MS_te=np.sum(1-np.abs(y_test-predict_NL_test)/y_test)
average_accuracy_tr = MS_tr/N_train
average_accuracy_te = MS_te/N_test
print(f'average_accuracy_tr_= {average_accuracy_tr}\naverage_accuracy_te = {average_accuracy_te}\n')

plt.show()