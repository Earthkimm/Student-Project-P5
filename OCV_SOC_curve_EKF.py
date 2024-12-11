import pandas as pd,  matplotlib.pyplot as plt,  numpy as np, scipy.stats as stats
from project_colors import *

#################################
#
#   CHOOSE WHICH PLOTS TO PLOT
#
#################################
Residual_plot = False
Polynomial_fit_plot = False

def poly_fit_OCV(order, index=None, color=None):
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
    
    # For Residual plot:
    residuals = filtered_ocv - poly_values
    
    if Residual_plot or Polynomial_fit_plot:
        plt.rc('font', weight='normal', size=12)    # Do this for nicer plots in project report
    # Residuals vs Fitted values plot
    if Residual_plot:
        if index < 2:
            row = 0
        else:
            row = 1
            axs[row, index%2].set_xlabel("Fitted Values [V]")
        axs[row, index%2].plot(poly_values, residuals, color=dark_green)
        axs[row, index%2].hlines(0, -2, 6, orange)
        axs[row, index%2].set_xlim(3.55, 4.05)
        axs[row, index%2].set_yticks(np.linspace(-0.02, 0.02, 5))
        axs[row, index%2].set_ylim(-0.02, 0.02)
        axs[row, index%2].set_title(f"p = {order}")
        axs[row, index%2].grid(True)
        if index%2 == 0:
            axs[row, index%2].set_ylabel("Residuals [V]")

    # Plot the data
    if Polynomial_fit_plot:
        ax.plot(filtered_soc, filtered_ocv, label='Raw Data', color=dark_green)
        ax.plot(filtered_soc, poly_values, label='Polynomial Fit', color=color, linestyle='--')
        xticks = np.linspace(0.2, 0.8, 7)
        ax.set_xticks(xticks, labels=[int(np.round(xticks[i]*100, 1)) for i in range(len(xticks))])
        ax.set_xlabel('SOC [%]')
        ax.set_ylabel('OCV [V]')
        ax.grid(True)
    
    return poly_values, poly_coefficients

plot = False
figs = []
names = []
if Residual_plot:
    fig1, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 6))
    figs.append(fig1)
    names.append("Figures_EKF/Residual_plot_EKF")
    colors = [None]*4
    plot = True
if Polynomial_fit_plot:
    fig2, ax = plt.subplots(1, figsize=(6, 3))
    figs.append(fig2)
    names.append("Figures_EKF/OCV_SOC_EKF")
    colors = [red, orange, light_green, purple]
    plot = True

    for i, order in enumerate([3, 8, 15, 24]):
        poly_fit_OCV(order, i, colors[i])
    for fig, name in zip(figs, names):
        fig.savefig(f'{name}.pdf', dpi=400, bbox_inches='tight')
    plt.show()