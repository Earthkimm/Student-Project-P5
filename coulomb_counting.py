import numpy as np, matplotlib.pyplot as plt
from LoadProfiles import loadprofiles
from project_colors import *

# This constant decides the amount of noise on the input
var_epsilon = 10000

np.random.seed(42069)   # Set seed for consistent plotting

#######################
#
#   DEFINE FUNCTIONS
#
#######################
def Couloumb_Counting(input_, initial_z, var_epsilon=0):
    maxIter = len(input_)
    z = initial_z
    zstore = np.zeros(maxIter)
    zstore[0] = z
    SigmaZstore = np.zeros(maxIter)
    for k in range(1, maxIter):
        z = z - delta_t/Q*input_[k-1]
        zstore[k] = z
        SigmaZ = (-delta_t/Q)**2 * k * var_epsilon
        SigmaZstore[k] = SigmaZ
    return maxIter, zstore, SigmaZstore

###########################
#
#   INITIALISE CONSTANTS
#
###########################
delta_t = 1
Q = 49.3*3600

###########################
#
#   INITIALISE VARIABLES
#
###########################
input_NoNoise = loadprofiles[2]

z_0 = 0.7
maxIter, ztrueStore = Couloumb_Counting(input_NoNoise, z_0)[:2]

noise = np.random.normal(0, np.sqrt(var_epsilon), maxIter)
input_Noise = input_NoNoise + noise
zhatStore, SigmaZstore = Couloumb_Counting(input_Noise, z_0, var_epsilon)[1:]

plt.figure(figsize=(6,3))
plt.plot(ztrueStore, color=orange)
plt.plot(zhatStore, color=dark_green)
plt.plot(zhatStore+3*np.sqrt(SigmaZstore), color=light_green)
plt.plot(zhatStore-3*np.sqrt(SigmaZstore), color=light_green)
plt.xlabel("Time [s]")
plt.xticks(np.linspace(0, 1400, 8))
plt.xlim(-5, 1400)
plt.ylabel("SOC [%]")
plt.yticks([0.6, 0.65, 0.7], [60, 65, 70])
plt.ylim(0.58, 0.73)
plt.grid()
plt.savefig("Figures_CC/CC_SOC_estimation.pdf", dpi=1000)
plt.show()