This Github contains the script for computing the simulation of SOC estimation using CC, LKF and EKF along with the code for computing the plots seen throughout the simulation sections of both CC, LKF and EKF. 

The file ‘Figures_CC’ contains the figures in chapter 5 regarding coulomb counting. The file ‘FIgures_LKF’ contains the figures in chapter 7 regarding linear kalman filtering. The file ‘FIgures_EKF’ contains the figures in chapter 8 regarding extended kalman filtering. 

The script 'Comparson_CC_LKF_EKF’ contains the code for comparing the two or more of the mentioned SOC estimation methods.  

The scripts ‘Noise_analysis_EKF.py’ and ‘Noise_analysis_LKF.py’ contains the code for comuting the noise analaysis plots for EKF and LKF, respectively.   

The script ‘LoadProfiles’ contains the script for visualising the speeds and loads of the different dynamic profiles.   

The scripts ‘OCV_SOC_curve_LKF.py’ and ‘OCV_SOC_curve_EKF.py’ contain the code for visualising the OCV-SOC data along with the linear fit/polynomial fits, respectively. 

 The scripts ‘extended_kalman_filter.py’ and ‘linear_kalman_filter.py’ contain the code for computing the simulations with 9/6 different noise combinations along with the code for computing the simulations with different load profiles. For varying the variance of the noise set the boolean ‘Varying_Sigmas’ to True and the boolean ‘Varying_inputs’ to false. To get the simulation with different load profiles set boolean ‘Varying_Sigmas’ to false and the boolean ‘Varying_inputs’ to true. 

The script ‘project_colors.py’ just contains the hex code for the colors used in the plots. 

Lastly the csv files ‘udds.csv’ and ‘us06.csv’ contain the data from dynamic profile 1 and 2. The file OCV_curve.csv contains OCV-SOC data.

 
