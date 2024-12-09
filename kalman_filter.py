import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_SOC_curve import a, b
from LoadProfiles import loadprofiles, profile_plots
from project_colors import *

###############################
#
#   CHOOSE PLOTTING STYLE
#
###############################
Varying_Sigmas = True
Varying_inputs = False
With_profileplots = Varying_inputs and False
Sigma_comparison = False

#######################
#
#   DEFINE FUNCTIONS
#
#######################
def Kalman_Filter(input_, measured_voltage, initial_x, initial_SigmaX, SigmaN, SigmaS):
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

def Plotting(ax, xstore, xhatstore, SigmaXstore, maxIter):
    ax.plot(xstore, color=orange, linestyle='-', label='True')
    ax.plot(xhatstore, color=dark_green, linestyle='-', label='Estimate')
    ax.plot(xhatstore+3*np.sqrt(SigmaXstore), color=light_green, linestyle='-.', label='Bounds')
    ax.plot(xhatstore-3*np.sqrt(SigmaXstore), color=light_green, linestyle='-.')
    ax.grid()
    ax.set_xticks(np.round(np.linspace(0, maxIter, 4), 0))
    ax.set_yticks([0.6, 0.65, 0.7, 0.75])#[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    ax.set_yticklabels([60, 65, 70, 75])#[5, 15, 25, 35, 45, 55, 65, 75]
    ax.set_ylim(0.55, 0.80)

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

###########################
#
#   INITIALISE VARIABLES
#
###########################
if Varying_Sigmas:
    inputs = loadprofiles[2:3]      # input is the same for all combinations of Sigmas (Dynamic Profile 1) and in a list
    SigmaN = [1e-1, 1e-4, 1e-7]     # Process-noise covariances
    SigmaS = [1e-2, 1e-1]     # Sensor-noise covariances
    
    fig, axs = plt.subplots(len(SigmaN), len(SigmaS), figsize=(9, 7))
elif Varying_inputs:
    inputs = loadprofiles[:]
    #inputs = [np.hstack([input]*6) for input in inputs]   # Used to test how the model reacts with longer inputs
    SigmaN = [1e-1] # Sigmas are constant when varying input
    SigmaS = [1e-2]

    titles = ["Constant 10A", "30A Pulses", "Dynamic Profile 1", "Dynamic Profile 2"]
    if With_profileplots:
        profile_plots(inputs, titles)

    limit = int(np.ceil(len(inputs)/2))
    fig, axs = plt.subplots(2, limit, sharey="row", figsize=(9, 7))
    axs = axs.reshape(1, len(inputs))
elif Sigma_comparison:
    inputs = loadprofiles[2:3]
    SigmaN = [1e-1, 1e-4, 1e-7]
    SigmaS = [1e-1, 1e-4, 1e-7]
    fig, axs = plt.subplots(max(len(SigmaN), len(SigmaS)), 2, sharex="col", sharey="row", figsize=(9, 7))

##############################################
#
#   DO THE SIMULATIONS AND CREATE THE PLOTS
#
##############################################
plot_col = 0    # This keeps track of what figure column is being worked with
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
            if Sigma_comparison:
                if plot_col == 0:
                    Sigma_n = SigmaN[1]
                    Sigma_s = SigmaS[plot_row]
                elif plot_col == 1:
                    Sigma_s = SigmaS[1]
                    Sigma_n = SigmaN[plot_row]
                else:
                    break
            # Initialise Kalman filter estimates and use the Kalman_Filter function to find lists of estimates
            xhat = np.array([[0.7],
                            [0]])
            SigmaX = np.ones((2, 2))
            xhatstore, SigmaXstore = Kalman_Filter(input_Noise, y_Noise, xhat, SigmaX, Sigma_n, Sigma_s)

            if Sigma_n == 1e-16:    # This is for testing if the variances are 0, they give errors in the kalman filter as the mathematical equations become unstable
                Sigma_n = 0
            if Sigma_s == 1e-16:
                Sigma_s = 0
            # Declare what axis to use for the current combination of Sigmas/inputs and plot in the given axis
            ax = axs[plot_row, plot_col]
            Plotting(ax, xstore[0], xhatstore[0], SigmaXstore[0], maxIter)
            
            # Determined by what kind of plot is chosen at the start of script make some final touches to the plot
            if Varying_Sigmas:
                if plot_col == 0:
                    ax.set_ylabel("SOC [%]")
                elif plot_col == len(SigmaS)-1:
                    twin = ax.twinx()
                    twin.set_ylabel("$\\hat{\\sigma}_{n}^2$="+format(Sigma_n, ".1e").replace('1.0e-0', '$10^{-').replace("0.0e+00", "${0")+"}$")#, rotation=-90)
                    twin.set_yticks([])
                    ax.set_yticklabels([])
                else:
                    ax.set_yticklabels([])
                if plot_row == 0:
                    ax.set_title("$\\hat{\\sigma}_{s}^2$="+format(Sigma_s, ".1e").replace('1.0e-0', '$10^{-').replace("0.0e+00", "${0")+"}$")
                    ax.set_xticklabels([])
                elif plot_row == 1:
                    ax.set_xticklabels([])
                elif plot_row == len(SigmaN)-1:
                    ax.set_xlabel("Time [s]")
            elif Varying_inputs:
                ax.set_title(titles[plot_col])
                if plot_col % limit == 0:
                    ax.set_ylabel("SOC [%]")
                if plot_col >= limit:
                    ax.set_xlabel("Time [s]")
            elif Sigma_comparison:
                twin = ax.twinx()
                twin.set_yticks([])
                if plot_col == 0:
                    twin.set_ylabel("$\\hat{\\sigma}_{s}^2$="+format(Sigma_s, ".1e").replace('1.0e-0', '$10^{-')+"}$")#, rotation=-90)
                    ax.set_ylabel("SOC [%]")
                elif plot_col == 1:
                    twin.set_ylabel("$\\hat{\\sigma}_{n}^2$="+format(Sigma_n, ".1e").replace('1.0e-0', '$10^{-')+"}$")#, rotation=-90)
                if plot_row == 0:
                    if plot_col == 0:
                        ax.set_title("$\\hat{\\sigma}_{n}^2$="+format(Sigma_n, ".1e").replace("1.0e-0", "$10^{-")+"}$")
                    elif plot_col == 1:
                        ax.set_title("$\\hat{\\sigma}_{s}^2$="+format(Sigma_s, ".1e").replace("1.0e-0", "$10^{-")+"}$")
                elif plot_row == len(axs)-1:
                    ax.set_xlabel("Time [s]")
                
            plot_row += 1   # Prepare for next combination
        plot_col += 1   # After every plot is made for a given column, go to the next
if Varying_Sigmas:
    plt.tight_layout(h_pad=-1)  # Adjust h_pad to avoid big gaps (between plots) created from tight_layout
    plt.savefig("Figurer/NoiseAnalysisSOC5.pdf", dpi=1000)
elif Varying_inputs:
    plt.tight_layout()
    plt.savefig("Figurer/LoadProfiles.pdf", dpi=1000)
elif Sigma_comparison:
    plt.tight_layout()
    plt.savefig("Figurer/NoiseAnalysisSOC3.pdf", dpi=1000)
plt.show()