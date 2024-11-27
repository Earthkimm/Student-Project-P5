import pandas as pd,  matplotlib.pyplot as plt,  numpy as np, scipy.stats as stats

def linear_fit_OCV():
    # Load the CSV file
    file_path = '.\\OCV_curve.csv'
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

    """ OOOOLLLLDDD  # Fit a linear model to the filtered data using numpy.polyfit
    coefficients = np.polyfit(filtered_soc, filtered_ocv, 1, full=True)
    #print(f"R^2: {1 - coefficients[1][0]/(np.sum((filtered_ocv - filtered_ocv.mean())**2))}")
    linear_fit = np.poly1d(coefficients[0])
    
    a = coefficients[0][0]
    b = coefficients[0][1]

    # Define the range for plotting
    #soc_range = np.linspace(0, 1, 100)
    ocv_fit = linear_fit(filtered_soc)
    residuals = filtered_ocv - ocv_fit
    #residuals = np.random.normal(1, 4, len(soc))
    #standardised_residuals = residuals/np.sqrt(np.sum(residuals**2)/(len(residuals)-1))"""
    
    # QQ plot
    """plt.rc('font', weight='normal', size=12)
    fig, ax = plt.subplots(1, figsize=(6, 3))
    (osm, osr), yeet = stats.probplot(filtered_ocv, dist="norm", plot=plt)
    line1, line2 = ax.get_lines()
    line1.set_color('#00916E')
    line2.set_color('#FCB97D')
    ax.grid()
    ax.set_title("", pad=-10)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.savefig('Figurer/QQ-plot Python.pdf', dpi=400, bbox_inches='tight')
    plt.show()
    
    # Residuals vs Fitted values plot
    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.scatter(fitted_values, residuals, color='#00916E')
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    #ax.set_title("Residual Plot")
    ax.hlines(0, -2, 6, '#FCB97D')
    ax.set_xlim(3.25, 4.25)
    ax.grid()
    plt.savefig('Figurer/Residual_plot Python.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    # Plot the data
    fig, ax = plt.subplots(1, figsize=(6, 3))
    ax.plot(soc, ocv, label='Raw Data', color='#00916E')
    ax.plot(filtered_soc, fitted_values, label='Linear Fit', color='#FCB97D', linestyle='--')
    xticks = ax.get_xticks()[1:-1]  # Save ticks to modify labels into %
    ax.set_xticks(xticks, labels=[np.round(xticks[i]*100, 1) for i in range(len(xticks))])
    ax.set_xlabel('SOC [%]')
    ax.set_ylabel('OCV [V]')
    #ax.set_title('OCV vs SOC at $25^\circ$C')
    ax.grid()
    ax.legend()
    plt.savefig('Figurer/OCV_SOC.pdf', format='pdf', dpi=400, bbox_inches='tight')
    plt.show()"""
    
    return beta_1[0], beta_2[0]

b, a = linear_fit_OCV()