import numpy as np, matplotlib.pyplot as plt, pandas as pd
from project_colors import *

# Choose to plot loadprofiles and speeds plot or not
LoadProfiles_WithSpeeds_plot = False

##########################
#
#   Create loadprofiles
#
##########################
constant_current = np.ones(1370)*10

pulse_currents = np.zeros(1370)
for i in range(6):
    interval = (i+1)*200
    pulse_currents[interval:interval+60] = 30

fp1 = "./udds.csv"
df1 = pd.read_csv(fp1)
dynamic_profile_1 = np.array(df1["Normalized current (A)"])
dynamic_profile_1 /= dynamic_profile_1[0]*2 # Under the assumption that a non-moving EV uses 0.5A the dataset is normalised according to this

fp2 = "./us06.csv"
df2 = pd.read_csv(fp2)
dynamic_profile_2 = np.array(df2["Normalized current (A)"])
dynamic_profile_2 /= dynamic_profile_2[0]*2

loadprofiles = [constant_current, pulse_currents, dynamic_profile_1, dynamic_profile_2]

# Only do the following if the plot is wanted
if LoadProfiles_WithSpeeds_plot:
    def profile_plots(loads, titles=["Constant 10A", "30A Pulses", "Dynamic Profile 1", "Dynamic Profile 2"], speeds=[pd.read_csv("./udds.csv")["Speed (km/h)"], pd.read_csv("./us06.csv")["Speed (km/h)"]]):
        # This function is used to create a plot of all the load profiles and the data has plots of their speeds aswell
        titles += titles[-2:]   # Since the last 2 profiles have 2 plots each, their titles are copied
        profiles = loads + speeds   # list of all data used to plot
        limit = int(np.ceil(len(loads)/2))  # This decides the amount of columns
        fig, axs = plt.subplots(3, limit, sharey="row", figsize=(9,12))
        axs = axs.reshape(1, len(profiles)) # Reshape the axes into a 1-D array for simpler access
        for profile in range(len(profiles)):
            ax = axs[0][profile]
            ax.plot(profiles[profile], color=dark_green)
            ax.set_title(titles[profile])
            ax.set_yticks(np.round(np.linspace(np.min(profiles[profile]), np.max(profiles[profile]), 4), -1))
            if profile % limit == 0 and profile <= limit:
                ax.set_ylabel("Load [A]")
            elif profile % limit == 0 and profile > limit:
                ax.set_ylabel("Speed [km/h]")
            if profile >= limit:
                ax.set_xlabel("Time [s]")
            ax.grid()
        plt.tight_layout()
        plt.savefig("Figures_LKF/LoadProfiles_WithSpeeds.pdf", dpi=1000)
        plt.show()
        return
    profile_plots(loadprofiles)