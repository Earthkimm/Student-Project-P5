import numpy as np, matplotlib.pyplot as plt, pandas as pd
from EKF_1 import EKF
from project_colors import *

# Kalder de forskellige data
ocv_curve = pd.read_csv(r".\\OCV_curve.csv")
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

    p, c = polfit(1, filtered_soc, filtered_ocv) #order of polynomial fit
    return c

c = linear_fit_OCV()
a, b = c[0], c[1] 

np.random.seed(42069)

dynamic_profile_NoNoise = udds["Normalized current (A)"]
dynamic_profile_NoNoise /= dynamic_profile_NoNoise[0]*2 # Under antagelsen, at når bilen ikke bevæger sig bruges 0.5A, omvendes normaliseringen

impulse_currents = np.zeros(1200)
for i in range(5):
    interval = (i+1)*200
    impulse_currents[interval:interval+60] = 30
constant_current = np.ones(1200)*10

inputs = [dynamic_profile_NoNoise]#[constant_current, impulse_currents, dynamic_profile_NoNoise]

delta_t = udds["Time (s)"][1] - udds["Time (s)"][0] # Skulle gerne være 1 s
R_1C_1 = 22
Q = 49.3*3600
R_1 = 0.0009
R_0 = 0.00127
z_0 = 0.7 #Initial guess - true is 70%


# Covariances for the noises
SigmaW = [1e-5]   # Process-noise covariance
SigmaV = [1e-3]   # Sensor-noise covariance 

# State-equation matrices
A = np.array([[1, 0],
              [0, np.exp(-delta_t / (R_1C_1))]])
B = np.array([[-delta_t / Q],
              [1-np.exp(-delta_t / (R_1C_1))]])
# Output-equation matrices
C = np.array([[a, -R_1]])
D = np.array([[-R_0]])


for input_NoNoise in inputs: # For hvert current-input kører vi igennem støjprofiler
    plot_row = 0    # Bruges til at holde styr på hvilken støjprofil vi har fat i
    maxIter = len(input_NoNoise)

    input_std_dev = (np.mean(input_NoNoise)*0.005/3)**2
    input_Noise = input_NoNoise + np.random.normal(0, np.sqrt(input_std_dev), maxIter)

    xtrue = np.array([[0.7],    # Initialize true system initial state
                      [0]])
    xstore = np.zeros((len(xtrue), maxIter+1))
    xstore[:,0] = xtrue.T[0]
    y_NoNoise = np.zeros(maxIter)
    for k in range(maxIter):
        y_NoNoise[k] = np.array([ocv_curve["OCV"][ocv_curve["SOC"] == np.round(xtrue[0, 0], 3)]]) + np.array([C[0, 1]*xtrue[1, 0]]) + np.dot(D, input_NoNoise[k])
        xtrue = np.matmul(A, xtrue) + B*input_NoNoise[k]
        xstore[:,k+1] = xtrue.T[0]
    y_std_dev = (np.mean(y_NoNoise)*0.005/3)**2
    #print(y_std_dev)

    y_Noise = y_NoNoise + np.random.normal(0, np.sqrt(y_std_dev), maxIter)
    for Sigma in SigmaV:    # For hver støjprofil laves det lineære kalmanfilter
        xhat = np.array([[z_0],     # Initialize Kalman filter initial estimate
                         [0]])
        SigmaX = np.ones((2, 2))    # Initialize Kalman filter covariance
        xhatstore = np.zeros((len(xhat), maxIter+1))
        xhatstore[:,0] = xhat.T[0]
        SigmaXstore = np.zeros((len(xhat)**2, maxIter+1))
        #_______________________________________
        xchat = np.array([[z_0],
                         [0]])
        xchatstore = np.zeros((len(xhat), maxIter+1))
        xchatstore[:,0] = xchat.T[0]
        #_______________________________________
        SigmaXc = np.zeros((2, 2))
        SigmaXcstore = np.zeros((len(xchat)**2, maxIter+1))

        for k in range(maxIter):
            # KF Step 1a: State-prediction time update
            xhat = np.matmul(A, xhat) + B*input_Noise[k]
            xchat = np.matmul(A, xchat) + B*input_Noise[k]
            SigmaXc = (delta_t / Q)**2*k*input_std_dev**2# np.matmul(np.matmul(A, SigmaXc),A.T) + SigmaW #(delta_t / Q)**2*k*input_std_dev**2 # ## #
            #print(SigmaXc)

            # KF Step 1b: Error-covariance time update
            SigmaX = np.matmul(np.matmul(A, SigmaX),A.T) + SigmaW
            ytrue = y_Noise[k]

            # KF Step 1c: Estimate system output
            yhat = np.matmul(C, xhat) + np.dot(D, input_Noise[k]) + b

            # KF Step 2a: Compute Kalman gain matrix
            SigmaY = np.matmul(np.matmul(C, SigmaX), C.T) + Sigma
            L = np.matmul(SigmaX, C.T)/SigmaY

            # KF Step 2b: State-estimate measurement update
            xhat += L*(ytrue - yhat)    # Grundet vores model ender vi med at plotte forskellen mellem lineære modellering og egentlig kurve i stedet for at xhat følger xtrue.

            # KF Step 2c: Error-covariance measurement update
            #joseph = np.eye(2) - np.matmul(L, C)
            #SigmaX = np.matmul(joseph, np.matmul(SigmaX, joseph.T)) + np.matmul(L, SigmaV*L.T)
            SigmaX -= np.matmul(np.matmul(L, SigmaY), L.T)
            """eigvals, eigvecs = np.linalg.eigh(SigmaX)
            eigvals[eigvals <= np.finfo(float).eps] = 0
            SigmaX = np.matmul(eigvecs, np.matmul(np.diag(eigvals), eigvecs.T))"""
            # [Store information for evaluation/plotting purposes]
            xchatstore[:,k+1] = xchat.T
            xhatstore[:,k+1] = xhat.T
            SigmaXstore[:,k+1] = SigmaX.flatten()
            SigmaXcstore[:,k+1] = SigmaXc.flatten()

        if Sigma > 0:
            for Sigmax in SigmaXstore.T:
                if np.all(np.linalg.eigvals(np.reshape(Sigmax, (2,2))) >= 0):
                    continue
                else:
                    print(f"{np.reshape(Sigmax, (2,2))} er ikke positiv definite")

        # Plot diverse current- og støjprofiler på deres respektive pladser

    #plot_col += 1   # Gør klar til at plotte næste current-input
#lt.tight_layout()
#plt.savefig("Figurer\\NoiseAnalysisSOC2.pdf", dpi=1000)
#plt.show()
plt.rc('font', weight='normal', size=16)
EE = EKF()

#---------------------------------------- PLOTS ------------------------------------------------------------

# 1) initial guess = 60% (change z_0)
# 2) initial guess = 70%
# 3) zoom-in med initial guess = 70%

# Comparison of all three methods
def CC_LKF_EKF_comparison(plot):
    plt.figure(figsize=(5, 4))
    plt.plot(xstore[0], color=orange, label='True')
    plt.plot(xchatstore[0], color=purple, linewidth = 2, linestyle = "dashed", label = "CC")#, linestyle = ":")
    plt.plot(xhatstore[0], color=dark_green, linewidth = 2, label="LKF")
    plt.plot(EE[0], color = red, label= "EKF", linestyle = "dashed")
    plt.grid()
    plt.ylabel("SOC [%]")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    if plot == 1:
        plt.xlim([-10,1400])
        plt.ylim([0.50,0.74])
        l = np.linspace(0.52,0.72,7)
        plt.yticks(l, labels=list(np.round(np.array(l)*100, 1)))
        plt.savefig(r".\CC_comp_60.pdf", dpi=1000)
    elif plot == 2:
        plt.xlim([-10,1400])
        plt.ylim([0.61,0.74])
        l = np.linspace(0.62,0.74,7)
        plt.yticks(l, labels=list(np.round(np.array(l)*100, 1)))
        plt.savefig(r".\CC_comp.pdf", dpi=1000)
    elif plot == 3:
        plt.xlim([750,850])
        plt.ylim([0.648,0.658])
        q = np.linspace(0.648,0.658,6)
        plt.yticks(q, labels = np.round(100*q,4))
        plt.savefig(r".\CC_comp_zoom.pdf", dpi=1000)
    plt.show()

# CC error # Difference between Coulomb Counting and True SOC
def CC_test():
    plt.figure(figsize=(4, 3))  # Adjust figure size
    plt.plot((xchatstore[0] - xstore[0])*100, color='#00916E')
    plt.ylabel("Residual SOC [%]")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.grid()
    plt.savefig(r".\\CC_2.pdf", dpi=1000)
    plt.show()

# Comparison of CC and LKF
def CC_LKF_comparison():
    plt.plot(xstore[0], color=orange, linestyle='-', label='True', linewidth = 2)
    plt.plot(xhatstore[0], color=dark_green, linewidth = 2, label="LKF")
    plt.plot(xchatstore[0], color=purple, linewidth = 2, label="CC", linestyle = "dashed") 
    plt.yticks([0.525, 0.550,0.575,0.600,0.625, 0.650, 0.675, 0.700, 0.725], labels=[52.5,55.0,57.5,60.0, 62.5, 65.0, 67.5, 70.0, 72.5])
    plt.yticks([0.625, 0.650, 0.675, 0.700, 0.725], labels=[62.5, 65.0, 67.5, 70.0, 72.5])
    #plt.plot(xchatstore[0] + 3*np.sqrt(SigmaXcstore[0]), color='#D4E4BC', linestyle='-', linewidth = 2)
    #plt.plot(xchatstore[0] - 3*np.sqrt(SigmaXcstore[0]), color='#D4E4BC', linestyle='-', linewidth = 2)
    #plt.xlim([-10,800])
    #plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5], labels=[-50, 0, 50, 100, 150])
    #plt.yticks([60,65,70])
    #plt.ylim([0.6,0.7])
    #plt.legend()
    plt.ylabel("SOC [%]")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.tight_layout()
    #plt.savefig(r".\\CCvslkf_1.pdf", dpi=1000)
    plt.show()

# Comparison of error between CC and LKF
def CC_LKF_comparison_error():
    plt.figure(figsize=(7, 3))
    plt.plot(xhatstore[0] - xstore[0], color=dark_green, linewidth = 2, label="LKF")
    plt.plot(xchatstore[0] - xstore[0], color=purple, linewidth = 2, label="CC") 
    plt.plot(xchatstore[0] + 3*np.sqrt(SigmaXcstore[0]), color=light_green, linestyle='-', linewidth = 2)
    plt.plot(xchatstore[0] - 3*np.sqrt(SigmaXcstore[0]), color=light_green, linestyle='-', linewidth = 2)
    plt.xlim([-10,800])
    plt.ylim([-0.02,0.04])
    plt.yticks([-0.02,-0.01,0.00,0.01,0.02,0.03], labels=[-2, -1,0,1,2,3])
    plt.ylabel("SOC [%]")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.tight_layout()
    #plt.savefig(r".\\CCvslkf_2_error.pdf", dpi=1000)
    plt.show()


CC_LKF_EKF_comparison(1)
#CC_LKF_comparison()
#CC_LKF_comparison_error()