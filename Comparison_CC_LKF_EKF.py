import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_SOC_curve import a, b
from LoadProfiles import loadprofiles, profile_plots
from project_colors import *

initial_SOC_guess = 0.6
include_CC = True
include_LKF = True
include_EKF = True
zoomed_in = False

#######################
#
#   DEFINE FUNCTIONS
#
#######################
def Couloumb_Counting(input_, initial_x, OCV_data, SOC_data):
    maxIter = len(input_)
    x = initial_x
    xstore = np.zeros((len(x), maxIter))
    xstore[:,0] = x.T[0]
    y = np.zeros(maxIter)
    for k in range(1, maxIter):
        x = np.matmul(A, x) + B*input_[k-1]
        y[k] = np.array(OCV_data[SOC_data == np.round(x[0, 0], 3)]) - R_1*x[1, 0] - R_0*input_[k]
        xstore[:,k] = x.T[0]
    return maxIter, xstore, y

def Linear_Kalman_Filter(input_, measured_voltage, initial_x, initial_SigmaX, SigmaN, SigmaS):
    maxIter = len(input_)
    xhat = initial_x
    SigmaX = initial_SigmaX
    xhatstore = np.zeros((len(xhat), maxIter))
    xhatstore[:,0] = xhat.T[0]
    SigmaXstore = np.zeros((len(xhat)**2, maxIter))
    SigmaXstore[:,0] = SigmaX.flatten()
    for k in range(1, maxIter):
        # KF Step 1a: State-prediction time update
        xhat = np.matmul(A, xhat) + B*input_[k-1]

        # KF Step 1b: Error-covariance time update
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T) + np.matmul(B*SigmaN,B.T)

        # KF Step 1c: Estimate system output
        yhat = np.matmul(C, xhat) + np.dot(D, input_[k]) + b

        # KF Step 2a: Compute Kalman gain matrix
        SigmaY = np.matmul(np.matmul(C, SigmaX), C.T) + SigmaS
        L = np.matmul(SigmaX, C.T)/SigmaY

        # KF Step 2b: State-estimate measurement update
        ytrue = measured_voltage[k]
        xhat += L*(ytrue - yhat)

        # KF Step 2c: Error-covariance measurement update
        SigmaX -= np.matmul(np.matmul(L, SigmaY), L.T)

        # [Store information for evaluation/plotting purposes]
        xhatstore[:,k] = xhat.T
        SigmaXstore[:,k] = SigmaX.flatten()
    return xhatstore, SigmaXstore

def poly_fit_OCV(order):
    # Load the CSV file
    file_path = './OCV_curve.csv'
    df = pd.read_csv(file_path)
    
    # Name SOC and OCV columns
    soc = df['SOC']
    ocv = df['OCV']
    
    # Define the range for filtering data
    min_soc = 0.2
    max_soc = 0.8

    # Filter the data within the specified SOC range
    mask = (soc >= min_soc) & (soc <= max_soc)
    filtered_soc = soc[mask]
    filtered_ocv = ocv[mask]
    
    # Create polynomial coefficients and reverse their order so the first in the array is for smallest order of z[k]
    poly_coefficients = np.polyfit(filtered_soc, filtered_ocv, order)[::-1]
    poly_values = np.zeros_like(filtered_soc)
    for i in range(order+1):
        poly_values += poly_coefficients[i] * filtered_soc ** i
    return poly_values, poly_coefficients

def C_(coef, x):
    cc1 = [i * coef[i] * x[0][0] ** (i-1) for i in range(len(coef))]
    return np.array([[sum(cc1), -R_1]])

def Extended_Kalman_Filter(poly_coef, input_, measured_voltage, initial_x, initial_SigmaX, SigmaN, SigmaS):
    maxIter = len(input_)
    xhat = initial_x
    SigmaX = initial_SigmaX
    xhatstore = np.zeros((len(xhat), maxIter))
    xhatstore[:,0] = xhat.T[0]
    SigmaXstore = np.zeros((len(xhat)**2, maxIter))
    for k in range(1, maxIter):
        # KF Step 1a: State-prediction time update
        xhat = np.matmul(A, xhat) + B*input_[k-1]

        # KF Step 1b: Error-covariance time update
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T) + np.matmul(B*SigmaN,B.T)

        # KF Step 1c: Estimate system output
        cc = [poly_coef[i] * xhat[0][0] ** (i-1) for i in range(len(poly_coef))]
        C = np.array([np.sum(cc), -R_1])
        yhat = np.matmul(C, xhat) + np.dot(D, input_[k])

        # KF Step 2a: Compute Kalman gain matrix
        C_hat = C_(poly_coef, xhat)
        SigmaY = np.matmul(np.matmul(C_hat, SigmaX), C_hat.T) + SigmaS
        L = np.matmul(SigmaX, C_hat.T)/SigmaY

        # KF Step 2b: State-estimate measurement update
        ytrue = measured_voltage[k]
        xhat += L*(ytrue - yhat)

        # KF Step 2c: Error-covariance measurement update
        SigmaX -= np.matmul(np.matmul(L, SigmaY), L.T)

        # [Store information for evaluation/plotting purposes]
        xhatstore[:,k] = xhat.T
        SigmaXstore[:,k] = SigmaX.flatten()
    return xhatstore, SigmaXstore

####################
#
#   LOAD DATASETS
#
####################
ocv_curve = pd.read_csv("./OCV_curve.csv")
OCV, SOC = ocv_curve["OCV"], ocv_curve["SOC"]
print(OCV[SOC == 0.002])    # Prints OCV when SOC = 0.002

fp = "./udds.csv"      # Used to access dataset for "Dynamic Profile 1"
df = pd.read_csv(fp)

###########################
#
#   INITIALISE CONSTANTS
#
###########################
delta_t = df["Time (s)"][1] - df["Time (s)"][0] # Gather timedifference from dataset (should be 1s)
R_1C_1 = 22
Q = 49.3*3600
R_1 = 0.0009
R_0 = 0.00127
# State-equation matrices
A = np.array([[1, 0],
              [0, np.exp(-delta_t / (R_1C_1))]])
B = np.array([[-delta_t / Q],
              [1-np.exp(-delta_t / (R_1C_1))]])
# Output-equation matrices
C = np.array([[a, -R_1]])
D = np.array([[-R_0]])
# Polynoimal fit
poly_coef = poly_fit_OCV(15)[1]

###########################
#
#   INITIALISE VARIABLES
#
###########################
SigmaN = [1e-1]
SigmaS = [1e-2]
inputs = loadprofiles[2:3]

##############################################
#
#   DO THE SIMULATIONS AND CREATE THE PLOTS
#
##############################################
for input_NoNoise in inputs:    # It is assumed inputs are without noise until it's added
    np.random.seed(42069)   # Set seed for consistent plotting
    
    # Initialise true system initial state and use CC to find voltage outputs of the battery
    xtrue = np.array([[0.7],
                        [0]])
    maxIter, xstore, y_NoNoise = Couloumb_Counting(input_NoNoise, xtrue, OCV, SOC)
    
    # Create noise for the current input
    input_std_dev = np.mean(input_NoNoise)*0.005/3
    noise = np.random.normal(0, input_std_dev, maxIter)
    input_Noise = input_NoNoise + noise
    
    # Create noise for the battery output
    y_std_dev = np.mean(y_NoNoise)*0.005/3
    y_Noise = y_NoNoise + np.random.normal(0, y_std_dev, maxIter)
    for Sigma_s in SigmaS:
        plot_row = 0    # Further specifies what combination of Sigmas/inputs are being worked with
        for Sigma_n in SigmaN:
            # Initialise Kalman filter estimates and use the different functions to find each of the estimates
            xhat = np.array([[initial_SOC_guess],
                            [0]])
            SigmaX = np.ones((2, 2))

            profiles = [xstore]
            colors = [orange]
            linestyles = ["-"]
            if not include_EKF:
                name = "Figures_LKF/"
            elif include_EKF:
                name = "Figures_EKF/"
            if include_CC:
                xhatstore_CC = Couloumb_Counting(input_Noise, xhat, OCV, SOC)[1]
                profiles.append(xhatstore_CC)
                colors.append(purple)
                linestyles.append("--")
                name += "CC_"
            if include_LKF:
                xhatstore_LKF = Linear_Kalman_Filter(input_Noise, y_Noise, xhat, SigmaX, Sigma_n, Sigma_s)[0]
                profiles.append(xhatstore_LKF)
                colors.append(dark_green)
                linestyles.append("-")
                name += "LKF_"
            if include_EKF:
                xhatstore_EKF = Extended_Kalman_Filter(poly_coef, input_Noise, y_Noise, xhat, SigmaX, Sigma_n, Sigma_s)[0]
                profiles.append(xhatstore_EKF)
                colors.append(red)
                linestyles.append("--")
                name += "EKF_"
            name += "comp_"

plt.rc('font', weight='normal', size=16)
plt.figure(figsize=(5, 4))
for i in range(len(profiles)):
    plt.plot(profiles[i][0], color=colors[i], linewidth=2, linestyle=linestyles[i])
plt.grid()
plt.ylabel("SOC [%]")
plt.xlabel("Time [s]")
if zoomed_in:
    plt.xlim([750,850])
    plt.ylim([0.648,0.658])
    ticks = np.linspace(0.648,0.658,6)
    plt.yticks(ticks, labels = np.round(100*ticks,4))
    name += "zoom"
else:
    shift_factor = 0.7-initial_SOC_guess
    plt.xlim([-10,1400])
    plt.ylim([0.62-shift_factor,0.74])
    ticks = np.linspace(0.62-shift_factor,0.74,7)
    plt.yticks(ticks, labels=[int(np.round(np.array(ticks[i])*100, 1)) for i in range(len(ticks))])
    name += f"{initial_SOC_guess*100:.0f}"
plt.tight_layout()
plt.savefig(name+".pdf", dpi=1000)
plt.show()