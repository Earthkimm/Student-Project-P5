import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_SOC_curve import a, b
from LoadProfiles import loadprofiles, profile_plots

np.random.seed(42069)   # Set seed for consistant plotting

###############################
#
#   CHOOSE PLOTTING STYLE
#
###############################
Varying_Sigmas = True
Varying_inputs = not Varying_Sigmas

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
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T) + SigmaW
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

def Plotting(ax, xstore, xhatstore, SigmaXstore, maxIter):
    ax.plot(xstore, color='#FCB97D', linestyle='-', label='True')
    ax.plot(xhatstore, color='#00916E', linestyle='-', label='Estimate')
    ax.plot(xhatstore+3*np.sqrt(SigmaXstore), color='#D4E4BC', linestyle='-.', label='Bounds')
    ax.plot(xhatstore-3*np.sqrt(SigmaXstore), color='#D4E4BC', linestyle='-.')
    ax.grid()
    ax.set_xticks(np.round(np.linspace(0, maxIter, 4), 0))
    ax.set_yticks([0.75, 0.7, 0.65, 0.6])
    ax.set_yticklabels([75, 70, 65, 60])
    ax.set_ylim(0.55, 0.80)

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
if Varying_Sigmas:
    inputs = loadprofiles[2:3]      # input is the same for all combinations of Sigmas (Dynamic Profile 1)
    SigmaW = [1e-3, 1e-5, 1e-7]     # Process-noise covariances
    SigmaV = [1e-3, 1e-2, 1e-1]     # Sensor-noise covariances
    
    fig, axs = plt.subplots(len(SigmaW), len(SigmaV), figsize=(9, 7))
    fig.delaxes(axs[2, 0])  # This axis is removed because the deviation of the estimate is too big
    subplots = ["(a)", "(b)", "(c)", "(d)\nTime [s]", "(e)", "(f)", "dummy", "(g)\nTime [s]", "(h)\nTime[s]"]
elif Varying_inputs:
    inputs = loadprofiles[:]
    SigmaW = [1e-5] # Sigmas are constant when varying input
    SigmaV = [1e-3]

    titles = ["Constant 10A", "30A Pulses", "Dynamic Profile 1", "Dynamic Profile 2"]
    profile_plots(inputs, titles)

    limit = int(np.ceil(len(inputs)/2))
    fig, axs = plt.subplots(2, limit, sharey="row", figsize=(9, 7))
    axs = axs.reshape(1, len(inputs))

##############################################
#
#   DO THE SIMULATIONS AND CREATE THE PLOTS
#
##############################################
plot_col = 0    # This keeps track of what figure column is being worked with
for input_NoNoise in inputs:    # It is assumed inputs are without noise until it's added
    # Initialise true system initial state and use CC to find voltage outputs of the battery
    xtrue = np.array([[0.7],
                        [0]])
    maxIter, xstore, y_NoNoise = Couloumb_Counting(input_NoNoise, xtrue, OCV, SOC)
    
    # Create noise for the current input
    input_std_dev = np.mean(input_NoNoise)*0.005/3
    input_Noise = input_NoNoise + np.random.normal(0, input_std_dev, maxIter)

    # Create noise for the battery output
    y_std_dev = np.mean(y_NoNoise)*0.005/3
    y_Noise = y_NoNoise + np.random.normal(0, y_std_dev, maxIter)
    for Sigma_V in SigmaV:
        plot_row = 0    # Further specifies what combination of Sigmas/inputs are being worked with
        for Sigma_W in SigmaW:
            # Initialise Kalman filter estimates and use the Kalman_Filter function to find lists of estimates
            xhat = np.array([[0.7],
                            [0]])
            SigmaX = np.ones((2, 2))
            xhatstore, SigmaXstore = Kalman_Filter(input_Noise, y_Noise, xhat, SigmaX, Sigma_W, Sigma_V)

            # Declare what axis to use for the current combination of Sigmas/inputs and plot in the given axis
            ax = axs[plot_row, plot_col]
            Plotting(ax, xstore[0], xhatstore[0], SigmaXstore[0], maxIter)
            
            # Determined by what kind of plot is chosen at the start of script make some final touches to the plot
            if Varying_Sigmas:
                ax.set_xlabel(subplots[3*plot_row + plot_col])
                if plot_col == 0:
                    ax.set_ylabel("SOC [%]")
                elif plot_col == len(SigmaV)-1:
                    twin = ax.twinx()
                    twin.set_ylabel("$\Sigma_{\\tilde{w}}$="+format(SigmaW[plot_row], ".1e").replace('1.0e-0', '$10^{-')+"}$")#, rotation=-90)
                    twin.set_yticks([])
                    ax.set_yticklabels([])
                elif plot_col == 1 and plot_row == len(SigmaW)-1:
                    ax.set_ylabel("SOC [%]")
                else:
                    ax.set_yticklabels([])
                if plot_row == 0:
                    ax.set_title("$\Sigma_{\\tilde{v}}$="+format(Sigma_V, ".1e").replace('1.0e-0', '$10^{-')+"}$")
                    ax.set_xticklabels([])
                elif plot_row == 1 and plot_col in [1, 2]:
                    ax.set_xticklabels([])
            elif Varying_inputs:
                ax.set_title(titles[plot_col])
                if plot_col % limit == 0:
                    ax.set_ylabel("SOC [%]")
                elif plot_col >= limit:
                    ax.set_xlabel("Time [s]")
            plot_row += 1   # Prepare for next combination
        plot_col += 1   # After every plot is made for a given column, go to the next
if Varying_Sigmas:
    plt.tight_layout(h_pad=-1)  # Adjust h_pad to avoid big gaps (between plots) created from tight_layout
    plt.savefig("Figurer\\NoiseAnalysisSOC5.pdf", dpi=1000)
elif Varying_inputs:
    plt.tight_layout()
    plt.savefig("Figurer\\LoadProfiles.pdf", dpi=1000)
plt.show()