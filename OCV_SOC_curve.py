import pandas as pd,  matplotlib.pyplot as plt,  numpy as np, scipy.stats as stats
from project_colors import *

#################################
#
#   CHOOSE WHICH PLOTS TO PLOT
#
#################################
QQ_plot = False
Residual_plot = False
Linear_fit_plot = False

def linear_fit_OCV():
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

    # Design matrix, X = [[z[1] 1],...,[z[n] 1]]^T
    X = np.array([np.ones_like(filtered_soc),
                  filtered_soc]).T
    
    # Vector of observed OCV values 
    Y = np.array([filtered_ocv[:]]).T
    
    # Beta = [a b]^T 
    # MLE of beta, beta_mle = (X^T*X)^{-1}*X^T*Y
    beta_mle = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)
    
    beta_1, beta_2 = beta_mle  
    
    # For QQ and residual plots:
    fitted_values = np.matmul(X, beta_mle)
    residuals = Y - fitted_values
    
    if QQ_plot or Residual_plot or Linear_fit_plot:
        plt.rc('font', weight='normal', size=12)    # Do this for nicer plots in project report
    # QQ plot
    if QQ_plot:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        (osm, osr), line_info = stats.probplot(filtered_ocv, dist="norm", plot=plt)
        line1, line2 = ax.get_lines()   # stats.probplot creates is own lines with colors, this is just to changes these colors
        line1.set_color(dark_green)
        line2.set_color(orange)
        ax.grid()
        ax.set_title("", pad=-10)   # stats.probplot creates a title for the plot, this is unwanted and is thus removed, pad is to counteract the spacing made for the title
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        plt.savefig('Figurer/QQ-plot Python.pdf', dpi=400, bbox_inches='tight')
        plt.show()
    
    # Residuals vs Fitted values plot
    if Residual_plot:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.scatter(fitted_values, residuals, color=dark_green)
        ax.set_xlabel("Fitted Values [V]")
        ax.set_ylabel("Residuals [V]")
        ax.hlines(0, -2, 6, orange)
        ax.set_xlim(3.25, 4.25)
        ax.grid()
        plt.savefig('Figurer/Residual_plot Python.pdf', dpi=400, bbox_inches='tight')
        plt.show()

    # Plot the data
    if Linear_fit_plot:
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.plot(soc, ocv, label='Raw Data', color=dark_green)
        ax.plot(filtered_soc, fitted_values, label='Linear Fit', color=orange, linestyle='--')
        xticks = ax.get_xticks()[1:-1]  # Save ticks to modify labels into %
        ax.set_xticks(xticks, labels=[np.round(xticks[i]*100, 1) for i in range(len(xticks))])
        ax.set_xlabel('SOC [%]')
        ax.set_ylabel('OCV [V]')
        ax.grid()
        plt.savefig('Figurer/OCV_SOC.pdf', format='pdf', dpi=400, bbox_inches='tight')
        plt.show()
    
    return beta_1[0], beta_2[0]

b, a = linear_fit_OCV()