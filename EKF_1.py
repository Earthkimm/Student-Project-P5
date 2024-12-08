import numpy as np, matplotlib.pyplot as plt, pandas as pd, matplotlib.colors as colors

# Load the CSV files
ocv_curve = pd.read_csv('.\\OCV_curve.csv')
udds = pd.read_csv(r".\\udds.csv") #".\\us06.csv"

def linear_fit_OCV():
    # Name SOC and OCV columns
    soc = ocv_curve['SOC']
    ocv = ocv_curve['OCV']
    
    # Define the range for filtering data
    min_soc = 0.2
    max_soc = 0.8

    # Filter the data within the specified SOC range
    mask = (soc >= min_soc) & (soc <= max_soc)
    filtered_soc = soc[mask]
    filtered_ocv = ocv[mask]

    def polfit(n, data_soc, data_ocv):
        coef = np.polyfit(data_soc, data_ocv, n)
        data1 = np.zeros_like(data_soc)
        for i in range(n+1):
            data1 += coef[n-i] * data_soc ** i
        return data1, coef

    p, c = polfit(15, filtered_soc, filtered_ocv) #order of polynomial fit
    return c

coef1 = linear_fit_OCV()
u3 = udds["Normalized current (A)"]

dynamic_profile = udds["Normalized current (A)"]
dynamic_profile /= dynamic_profile[0]*2 # Under antagelsen, at når bilen ikke bevæger sig bruges 0.5A, omvendes normaliseringen


impulse_currents = np.zeros(1200)
for i in range(5):
    interval = (i+1)*200
    impulse_currents[interval:interval+60] = 30
constant_current = np.ones(1200)*10

inputs = [dynamic_profile]#[constant_current]#, impulse_currents]#, dynamic_profile]

delta_t = udds["Time (s)"][1] - udds["Time (s)"][0] # Skulle gerne være 1 s
R_1C_1 = 22
Q = 49.3*3600
R_1 = 0.0009
R_0 = 0.00127

# Initialize simulation variables
SigmaW = [10**-3]#[10**-7, 10**-3, 10**-2]#[10**-10, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1]#, (np.mean(dynamic_profile)*0.005/3)**2]    # Process-noise covariance
SigmaV = [10**-4]#[10**-4, 10**-3, 10**-2]#[10**-10, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 5e-1]  # Sensor-noise covariance (Er sat meget lavt lige nu for at vise hvordan at xhat IKKE følger xtrue)
# State-equation matrices
A = np.array([[1, 0], [0, np.exp(-delta_t / (R_1C_1))]])
B = np.array([[-delta_t / Q, 1 - np.exp(-delta_t / (R_1C_1))]]).T
D = np.array([[1]])

# Function to calculate C matrix
def C_(x):
    cc1 = [i * coef1[len(coef1)-1-i] * x[0][0] ** (i-1) for i in range(len(coef1))]
    return np.array([[sum(cc1), -R_1]])

def input_1(xtrue_):
    i_u = np.array(u3)
    y_u = []
    for k in range(len(i_u)):
        xtrue_ = np.matmul(A, xtrue_) + B * i_u[k]
        ytrue = np.array([ocv_curve["OCV"][ocv_curve["SOC"] == np.round(xtrue_[0, 0], 3)]]) - R_1 * xtrue_[1, 0] - R_0 * i_u[k]
        y_u.append(ytrue[0][0])
    
    Sigma_n = (np.mean(dynamic_profile) * 0.005 / 3) ** 2
    Sigma_s = (np.mean(y_u) * 0.005 / 3) ** 2

    i_m = i_u + np.random.normal(0,np.sqrt(Sigma_n), len(i_u))
    y_m = np.array(y_u) + np.random.normal(0,np.sqrt(Sigma_s),len(i_u))

    return i_m, y_m, Sigma_n, Sigma_s

i_m, y_m, Sigma_n, Sigma_s = input_1(np.array([[0.7],[0]]))



# Nok plots til støj- og inputprofiler
RMSE_matrix = np.zeros((len(SigmaW), len(SigmaV)))
bound_width_store = np.zeros((len(SigmaW), len(SigmaV), len(inputs[0])+1))
bound_width_mean_store = np.zeros_like(RMSE_matrix)
reliability_matrix = np.zeros_like(RMSE_matrix)

for input in inputs:    # For hvert current-input kører vi igennem støjprofiler
    col = 0
    L_xhat = []
    L_x = []
    Sigma_x11 = []
    temp11 = []
    for Sigma_V in SigmaV:
        row = 0
        for Sigma_W in SigmaW:    # For hver støjprofil laves det lineære kalmanfilter
            xtrue = np.array([[0.7],    # Initialize true system initial state
                            [0]])
            xhat = np.array([[0.6],     # Initialize Kalman filter initial estimate#___________________________________________________________________________
                            [0]])
            SigmaX = np.ones((2, 2))    # Initialize Kalman filter covariance
            
            maxIter = len(input)
            xstore = np.zeros((len(xtrue), maxIter+1))
            xstore[:,0] = xtrue.T[0]
            xhatstore = np.zeros((len(xhat), maxIter+1))
            xhatstore[:,0] = xhat.T[0]
            SigmaXstore = np.zeros((len(xhat)**2, maxIter+1))
            SigmaXstore[:,0] = SigmaX.flatten()

            
            if np.abs(xtrue[0]-xhat[0]) > 3*np.sqrt(SigmaX[0, 0]):
                    reliability_matrix[0, 0] += 1
            
            L_xhat_temp = [xhat[0][0]]
            L_x_temp = [xtrue[0][0]]
            Sigma_x1 = [SigmaX[0][0]]
            temp1 = []
            for k in range(maxIter):
                #w = np.random.normal(0, np.sqrt(Sigma_W), (1, len(xtrue)))
                xhat = np.matmul(A, xhat) + B * i_m[k]# + w.T
                
                C = C_(xhat)
                SigmaX = np.matmul(np.matmul(A, SigmaX), A.T) + np.matmul(B, Sigma_W * B.T)

                #v = np.random.normal(0, np.sqrt(Sigma_V), (1, len(np.matmul(C, xtrue))))
                #ytrue = np.array([ocv_curve["OCV"][ocv_curve["SOC"] == np.round(xtrue[0, 0], 3)]]) - R_1 * xtrue[1, 0] - R_0 * input[k] + v
                xtrue = np.matmul(A, xtrue) + B * input[k] 

                cc = [coef1[len(coef1)-1-i] * xhat[0][0] ** i for i in range(len(coef1))]
                yhat = sum(cc) - R_1 * xhat[1][0] - R_0 * i_m[k]

                SigmaY = np.matmul(np.matmul(C, SigmaX), C.T) + Sigma_V
                L = np.matmul(SigmaX, C.T) / SigmaY
                xhat += L * (y_m[k] - yhat)
                
                L_xhat_temp.append(xhat[0][0])
                L_x_temp.append(xtrue[0][0])
                Sigma_x1.append(SigmaX[0][0])
                temp1.append(L[0][0])

                SigmaX -= np.matmul(np.matmul(L, SigmaY), L.T)
                xstore[:,k+1] = xtrue.T
                xhatstore[:,k+1] = xhat.T
                SigmaXstore[:,k+1] = SigmaX.flatten()
                #q = np.eye(2) - np.matmul(L, C)
                #SigmaX = np.matmul(np.matmul(q, SigmaX), q.T) + np.matmul((L * Sigma_V), L.T)
            L_xhat.append(L_xhat_temp)
            L_x.append(L_x_temp)
            Sigma_x11.append(Sigma_x1)
            temp11.append(temp1)
            
            
            # Store results

            
            
            
            reliability_matrix[row,col] = np.sum(np.abs(xstore[0]-xhatstore[0])>3*np.sqrt(SigmaXstore[0]))/(maxIter+1)
            RMSE_matrix[row, col] = np.sqrt(np.sum(np.abs(xstore[0,:]-xhatstore[0,:])**2)/(maxIter+1))
            bound_width_store[row, col] = 6*np.sqrt(SigmaXstore[0,0:])
            bound_width_mean_store[row, col] = np.mean(bound_width_store[row, col])
            row += 1
        col += 1

bound_width_mean_store = np.nan_to_num(bound_width_mean_store, nan=1e-10)
reliability_matrix[reliability_matrix == 0] = 1e-10

#print((np.array(temp11[1]))[1000:1100])

def EKF():
    return L_xhat

#L_xhat1 = np.reshape(L_xhat, (3,3, 1370))
#L_x1 = np.reshape(L_x, (3,3, 1370))
#Sigma_x111 = np.reshape(Sigma_x11, (3,3, 1370))
#temp111 = np.reshape(temp11, (3,3, 1370))


