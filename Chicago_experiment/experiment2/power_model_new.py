import numpy as np
import matplotlib.pyplot as plt
import requests
import json

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


    return average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5, frequency_CPU_6, \
           frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5

# prepare training data
average_frequency, Total_utilization, power, average_CPU_T,frequency_CPU_5,frequency_CPU_6,frequency_CPU_7, CPU_0_5_average_U, CPU_6_average_U, CPU_7_average_U, CPU_0_2_5 = data_prepocessing(original_data)
N_train = Total_utilization.shape[0]

CPU_0_average_U, CPU_1_average_U, CPU_1_average_U,CPU_1_average_U,CPU_1_average_U,CPU_1_average_U = CPU_0_2_5

X_train = np.array([Total_utilization.reshape(N_train,-1),
                    average_CPU_T.reshape(N_train,-1),np.ones((N_train,1))])

X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
                    average_CPU_T.reshape(N_train,-1),np.ones((N_train,1))])

X_train = np.array([Total_utilization.reshape(N_train,-1), CPU_0_5_average_U.reshape(N_train,-1), CPU_6_average_U.reshape(N_train,-1), CPU_7_average_U.reshape(N_train,-1),
                    CPU_0_average_U.reshape(N_train,-1), CPU_1_average_U.reshape(N_train,-1), CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),CPU_1_average_U.reshape(N_train,-1),np.ones((N_train,1))])


X_train = X_train.transpose(0,2,1).reshape((X_train.shape[0], N_train))
Train_X = X_train.transpose()
train_y = power.reshape(power.shape[0], 1)
M_train = Train_X.shape[1]



# prepare test data
average_frequency_test, Total_utilization_test, power_test, average_CPU_T_test, frequency_CPU_5_test,frequency_CPU_6_test,frequency_CPU_7_test, CPU_0_5_average_U_test, CPU_6_average_U_test, CPU_7_average_U_test, CPU_0_2_5_test = data_prepocessing(test_data)
N_test = Total_utilization_test.shape[0]

CPU_0_average_U_test, CPU_1_average_U_test, CPU_1_average_U_test,CPU_1_average_U_test,CPU_1_average_U_test,CPU_1_average_U_test = CPU_0_2_5_test


X_test = np.array([Total_utilization_test.reshape(N_test,-1),
                   average_CPU_T_test.reshape(N_test,-1),np.ones((N_test,1))])

X_test = np.array([Total_utilization_test.reshape(N_test,-1), CPU_0_5_average_U_test.reshape(N_test,-1), CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1),
                   average_CPU_T_test.reshape(N_test,-1),np.ones((N_test,1))])

X_test = np.array([Total_utilization_test.reshape(N_test,-1), CPU_0_5_average_U_test.reshape(N_test,-1), CPU_6_average_U_test.reshape(N_test,-1), CPU_7_average_U_test.reshape(N_test,-1),
                   CPU_0_average_U_test.reshape(N_test,-1), CPU_1_average_U_test.reshape(N_test,-1), CPU_1_average_U_test.reshape(N_test,-1),CPU_1_average_U_test.reshape(N_test,-1),CPU_1_average_U_test.reshape(N_test,-1),CPU_1_average_U_test.reshape(N_test,-1), np.ones((N_test,1))])


X_test = X_test.transpose(0,2,1).reshape((X_test.shape[0], N_test))
Test_X = X_test.transpose()
test_y = power_test.reshape(power_test.shape[0], 1)
M_test = Test_X.shape[1]


# Ordinary Linear Regression
w=np.zeros((M_train,1))
w=np.linalg.pinv(Train_X).dot(train_y) # pinv(A)*b
Predict_y_OLS=Train_X.dot(w)

np.savetxt('Original.txt', w)

# plot the curve of OLS
plt.figure(1)
plt.plot(train_y, 'go-', linewidth=1.5)
plt.plot(Predict_y_OLS, 'r*-', linewidth=1.5)
plt.xlabel("Training Sample")
plt.ylabel("Prediction")
plt.title("Ordinary Lineary Regression")
plt.legend(['Raw Data', 'Regreassion'])

#validation
# w=np.zeros((M_test,1))
# w=np.linalg.pinv(Test_X).dot(test_y) # pinv(A)*b
Predict_y_OLS_test=Test_X.dot(w)

plt.figure(2)
plt.plot(test_y, 'go-', linewidth=1.5)
plt.plot(Predict_y_OLS_test, 'r*-', linewidth=1.5)
plt.xlabel("Test Sample")
plt.ylabel("Prediction")
plt.title("Ordinary Lineary Regression")
plt.legend(['Raw Data', 'Regreassion'])

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
plt.plot(train_y, 'go-', linewidth=1.5)
plt.plot(predict_y_Local, 'b-*', linewidth=1.5) # Predictions in blue color.
plt.title("Locally Weighted Linear regression")
plt.xlabel("Training Sample")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'Regression'])

#validation
predict_y_Local_test = []
tau= 2
# Predicting for all nval values and storing them in preds.
for point in range(N_test):
    x_point=Test_X[point,:].reshape((1,M_test))
    w, pred = predict(Train_X, train_y, x_point, tau)
    predict_y_Local_test.append(pred)
# change prediction type and reshaping
predict_y_Local_test=np.array(predict_y_Local_test)
predict_y_Local_test=predict_y_Local_test.reshape((predict_y_Local_test.shape[0],-1))
print(predict_y_Local_test.shape)
print(test_y.shape)

plt.figure(4)
plt.plot(test_y, 'go-', linewidth=1.5)
plt.plot(predict_y_Local_test, 'b-*', linewidth=1.5) # Predictions in blue color.
plt.title("Locally Weighted Linear regression")
plt.xlabel("Test Sample")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'Regression'])

plt.figure(5)
plt.plot(train_y, 'go-', linewidth=1.5)
plt.plot(Predict_y_OLS, 'r*-', linewidth=1.5)
plt.plot(predict_y_Local, 'b-*', linewidth=1.5) # Predictions in blue color.
plt.title("OLR and LWLR Training Results Comparison")
plt.xlabel("Training Sample")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'OLR', 'LWLR'])

plt.figure(6)
plt.plot(test_y, 'go-', linewidth=1.5)
plt.plot(Predict_y_OLS_test, 'r*-', linewidth=1.5)
plt.plot(predict_y_Local_test, 'b-*', linewidth=1.5) # Predictions in blue color.
plt.title("OLR and LWLR Validation Results Comparison")
plt.xlabel("Test Sample")
plt.ylabel("Prediction")
plt.legend(['Raw Data', 'OLR', 'LWLR'])
print(w)
plt.show()
