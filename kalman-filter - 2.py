import numpy as np, matplotlib.pyplot as plt, pandas as pd, matplotlib.colors as colors
from OCV_SOC_curve import a, b
from LoadProfiles import loadprofiles

np.random.seed(42069)

#######################
#
#   DEFINE FUNCTIONS
#
#######################
def Kalman_Filter(input, measured_voltage, initial_x, initial_SigmaX, SigmaW, SigmaV):
    maxIter = len(input)
    xhat = initial_x
    SigmaX = initial_SigmaX
    xhatstore = np.zeros((len(xhat), maxIter+1))
    xhatstore[:,0] = xhat.T[0]
    SigmaXstore = np.zeros((len(xhat)**2, maxIter+1))
    for k in range(maxIter):
        # KF Step 1a: State-prediction time update
        xhat = np.matmul(A, xhat) + B*input_Noise[k]

        # KF Step 1b: Error-covariance time update
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T) + np.eye(2)*SigmaW
        ytrue = measured_voltage[k]

        # KF Step 1c: Estimate system output
        yhat = np.matmul(C, xhat) + np.dot(D, input_Noise[k]) + b

        # KF Step 2a: Compute Kalman gain matrix
        SigmaY = np.matmul(np.matmul(C, SigmaX), C.T) + SigmaV
        L = np.matmul(SigmaX, C.T)/SigmaY

        # KF Step 2b: State-estimate measurement update
        xhat += L*(ytrue - yhat)

        # KF Step 2c: Error-covariance measurement update
        SigmaX -= np.matmul(np.matmul(L, SigmaY), L.T)

        # [Store information for evaluation/plotting purposes]
        xhatstore[:,k+1] = xhat.T
        SigmaXstore[:,k+1] = SigmaX.flatten()
    return xhatstore, SigmaXstore

def Couloumb_Counting(input, initial_x, OCV_data, SOC_data):
    maxIter = len(input)
    xtrue = initial_x
    xstore = np.zeros((len(xtrue), maxIter+1))
    xstore[:,0] = xtrue.T[0]
    y_NoNoise = np.zeros(maxIter)
    for k in range(maxIter):
        y_NoNoise[k] = np.array([OCV_data[SOC_data == np.round(xtrue[0, 0], 3)]]) - R_1*xtrue[1, 0] - R_0*input[k]
        xtrue = np.matmul(A, xtrue) + B*input[k]
        xstore[:,k+1] = xtrue.T[0]
    return maxIter, xstore, y_NoNoise

def Colormesh(plot, matrix, vmin, cticks, xticks, yticks, cmap, extra=False):
    if not extra:
        extra = plot+" [%]"
    plt.pcolormesh(matrix, norm=colors.LogNorm(vmin=vmin), cmap=cmap)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(j+0.5, i+0.5, f"{matrix[i, j]*100:.3f}", color="white", ha="center", va="center")
    cbar = plt.colorbar(label=extra, ticks=cticks)
    cbar.set_ticklabels(np.round(cbar.get_ticks()*100, 1))
    plt.xticks(xticks[0], xticks[1])
    plt.xlabel("$\Sigma_{\\tilde{v}}$")
    plt.yticks(yticks[0], yticks[1])
    plt.ylabel("$\Sigma_{\\tilde{w}}$")
    plt.tight_layout()
    plt.savefig(f"Noise comparison colormeshes\\{plot} Colormesh.pdf", dpi=1000)
    plt.clf()

####################
#
#   LOAD DATASETS
#
####################
ocv_curve = pd.read_csv(".\\OCV_curve.csv")
OCV, SOC = ocv_curve["OCV"], ocv_curve["SOC"]
print(OCV[SOC == 0.002])    # Prints OCV when SOC = 0.002

fp = ".\\udds.csv"      # Used to access dataset for "Dynamic Profile 1"
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

###########################
#
#   INITIALISE VARIABLES
#
###########################
SigmaW = [0, 1e-7, 1e-6, 1e-5, 1e-4, 2.2e-4, 1e-3, 1e-2, 1e-1]  # Process-noise covariances
SigmaV = [0, 1e-7, 1e-6, 1e-5, 4.1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # Sensor-noise covariances
input = loadprofiles[2]

size = [len(SigmaW), len(SigmaV)]
RMSE_matrix = np.zeros(size)
bound_width_store = np.zeros(size + [len(input)+1])
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
maxIter, xstore, y_NoNoise = Couloumb_Counting(input, xtrue, OCV, SOC)

# Create noise for the current input
input_std_dev = np.mean(input)*0.005/3
input_Noise = input + np.random.normal(0, input_std_dev, maxIter)

# Create noise for the battery output
y_std_dev = np.mean(y_NoNoise)*0.005/3
y_Noise = y_NoNoise + np.random.normal(0, y_std_dev, maxIter)

##############################################
#
#   DO THE SIMULATIONS AND SAVE THE RESULTS
#
##############################################
col = 0
for Sigma_V in SigmaV:
    row = 0
    for Sigma_W in SigmaW:
        # Initialise Kalman filter estimates and use the Kalman_Filter function to find lists of estimates
        xhat = np.array([[0.7],
                        [0]])
        SigmaX = np.ones((2, 2))
        xhatstore, SigmaXstore = Kalman_Filter(input_Noise, y_Noise, xhat, SigmaX, Sigma_W, Sigma_V)

        RMSE_matrix[row, col] = np.sqrt(np.sum(np.abs(xstore[0,:]-xhatstore[0,:])**2)/(maxIter+1))
        bound_width_store[row, col] = 6*np.sqrt(SigmaXstore[0,0:])
        bound_width_mean_store[row, col] = np.mean(bound_width_store[row, col])
        reliability_matrix[row, col] = np.sum(np.abs(xstore[0]-xhatstore[0])>3*np.sqrt(SigmaXstore[0]))/(maxIter+1)
        
        row += 1
    col += 1

# Remove nan and 0 from the matrices as they bug out the plots
bound_width_mean_store = np.nan_to_num(bound_width_mean_store, nan=1e-10)
reliability_matrix[reliability_matrix == 0] = 1e-10

# Create a custom colormap with colors related to the project report
custom_cmap = cmap=colors.LinearSegmentedColormap.from_list("my_custom_cmap", ["#5B3758", "#00916E", "#D4E4BC", "#FCB97D"])

############################
#
#   CREATE AND SAVE PLOTS
#
############################
plt.rc('font', weight='normal', size=18)
plt.figure(figsize=(14, 6))
xticks = [[i+0.5 for i in range(len(SigmaV))],
          [format(Sigma_V, ".1e").replace("0.0e+00", "${0").replace("1.0e-0", "$10^{-").replace("e-0", "$\\cdot10^{-")+"}$" for Sigma_V in SigmaV]]
yticks = [[i+0.5 for i in range(len(SigmaW))],
          [format(Sigma_W, ".1e").replace("0.0e+00", "${0").replace("1.0e-0", "$10^{-").replace("e-0", "$\\cdot10^{-")+"}$" for Sigma_W in SigmaW]]

Colormesh("RMSE", RMSE_matrix, 0.01, [0.01, 0.011, 0.012, 0.013, 0.014, 0.015], xticks, yticks, custom_cmap)
Colormesh("Bound Width Error Mean", bound_width_mean_store, 0.01, [0.01, 0.1, 1], xticks, yticks, custom_cmap, "Error bound width mean [%]")
Colormesh("Deviation Percentage", reliability_matrix, 0.001, [0.001, 0.01, 0.1, 0.9], xticks, yticks, custom_cmap, "% of true SOC outside error bounds")