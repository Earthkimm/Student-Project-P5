import numpy as np, matplotlib.pyplot as plt, pandas as pd, matplotlib.colors as colors
from LoadProfiles import loadprofiles
from project_colors import *

np.random.seed(42069)

#######################
#
#   DEFINE FUNCTIONS
#
#######################
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
    SigmaXstore[:,0] = SigmaX.flatten()
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
        Joseph = np.eye(2) - np.matmul(L, C_hat)
        SigmaX = np.matmul(Joseph, np.matmul(SigmaX, Joseph.T)) + np.matmul(L, SigmaS * L.T)
        if not np.all(np.linalg.eigvals(SigmaX) >= 0):
            U, S, Vt = np.linalg.svd(SigmaX)
            H = np.matmul(Vt.T, np.matmul(S, Vt))
            SigmaX = (SigmaX + SigmaX.T + H + H.T)/4

        # [Store information for evaluation/plotting purposes]
        xhatstore[:,k] = xhat.T
        SigmaXstore[:,k] = SigmaX.flatten()
    return xhatstore, SigmaXstore

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

def Colormesh(plot, matrix, vmin, cticks, xticks, yticks, cmap, extra=False, vmax=None, roundfactor=1):
    if not extra:
        extra = plot+" [%]"
    plt.pcolormesh(matrix, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(j+0.5, i+0.5, f"{matrix[i, j]*100:.3f}", color="white", ha="center", va="center")
    cbar = plt.colorbar(label=extra, ticks=cticks)
    cbar.set_ticklabels(np.round(np.array(cticks)*100, roundfactor))
    plt.xticks(xticks[0], xticks[1])
    plt.xlabel("$\\hat{\sigma}_{s}^2$")
    plt.yticks(yticks[0], yticks[1])
    plt.ylabel("$\\hat{\sigma}_{n}^2$", rotation=0)
    plt.tight_layout()
    plt.savefig(f"Figures_EKF/Noise_analysis_{plot}_EKF.pdf", dpi=1000)
    plt.clf()

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
D = np.array([[-R_0]])
# Polynoimal fit
poly_coef = poly_fit_OCV(15)[1]

###########################
#
#   INITIALISE VARIABLES
#
###########################
SigmaN = [1e-16, 1e-7, 1e-6, 1e-5, 1e-4, 2.2e-4, 1e-3, 1e-2, 1e-1]  # Process-noise covariances [0, 1e+3, 1e+4, 1e+5, 1e+6, 1e+7, 2.2e+7, 1e+8, 1e+9]#
SigmaS = [1e-16, 1e-7, 1e-6, 1e-5, 4.1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Sensor-noise covariances [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]#
input_NoNoise = loadprofiles[2]

size = [len(SigmaN), len(SigmaS)]
RMSE_matrix = np.zeros(size)
bound_width_store = np.zeros(size + [len(input_NoNoise)])
bound_width_mean_store = np.zeros(size)
reliability_matrix = np.zeros(size)

#############################################
#
#   INITIALISE INPUT AND OUTPUT OF BATTERY
#
#############################################
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

##############################################
#
#   DO THE SIMULATIONS AND SAVE THE RESULTS
#
##############################################
col = 0
for Sigma_s in SigmaS:
    row = 0
    for Sigma_n in SigmaN:
        # Initialise Kalman filter estimates and use the Kalman_Filter function to find lists of estimates
        xhat = np.array([[0.7],
                        [0]])
        SigmaX = np.ones((2, 2))
        xhatstore, SigmaXstore = Extended_Kalman_Filter(poly_coef, input_Noise, y_Noise, xhat, SigmaX, Sigma_n, Sigma_s)

        RMSE_matrix[row, col] = np.sqrt(np.sum(np.abs(xstore[0,:]-xhatstore[0,:])**2)/(maxIter))
        bound_width_store[row, col] = 6*np.sqrt(SigmaXstore[0,0:])
        bound_width_mean_store[row, col] = np.mean(bound_width_store[row, col])
        reliability_matrix[row, col] = np.sum(np.abs(xstore[0]-xhatstore[0])>3*np.sqrt(SigmaXstore[0]))/(maxIter)
        
        row += 1
    col += 1

# Remove nan and 0 from the matrices as they bug out the plots
SigmaN[0] = 0; SigmaS[0] = 0    # Set to zero for plotting
bound_width_mean_store = np.nan_to_num(bound_width_mean_store, nan=1e-10)
reliability_matrix[reliability_matrix == 0] = 1e-10

# Create a custom colormap with colors related to the project report
custom_cmap = cmap=colors.LinearSegmentedColormap.from_list("my_custom_cmap", [purple, dark_green, light_green, orange])

############################
#
#   CREATE AND SAVE PLOTS
#
############################
plt.rc('font', weight='normal', size=18)
plt.figure(figsize=(14, 6))
xticks = [[i+0.5 for i in range(len(SigmaS))],
          [format(Sigma_s, ".1e").replace("0.0e+00", "${0").replace("1.0e-0", "$10^{-").replace("e-0", "$\\cdot10^{-")+"}$" for Sigma_s in SigmaS]]
yticks = [[i+0.5 for i in range(len(SigmaN))],
          [format(Sigma_n, ".1e").replace("0.0e+00", "${0").replace("1.0e-0", "$10^{-").replace("e-0", "$\\cdot10^{-")+"}$" for Sigma_n in SigmaN]]

Colormesh("RMSE", RMSE_matrix, 0.0004, [0.0004, 0.0006, 0.001], xticks, yticks, custom_cmap, roundfactor=2)
Colormesh("EBW_mean", bound_width_mean_store, 0.001, [0.001, 0.01, 0.09], xticks, yticks, custom_cmap, "Error bound width mean [%]")
Colormesh("Percentage_outside_EB", reliability_matrix, 0.001, [0.001, 0.01, 0.1, 0.9], xticks, yticks, custom_cmap, "% of true SOC outside error bounds")