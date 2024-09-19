import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_dir = "output"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

line1 = pd.read_csv("line_data.txt",sep="\s+",header=None)
line2 = pd.read_csv("line_data_2.txt",sep="\s+",header=None)


# Homogeneous least squares fit function
def homogeneous_least_squares(x, y):
    # Assemble the data matrix for homogeneous system
    A = np.vstack([x, y, np.ones(len(x))]).T

    # Singular value decomposition (SVD) to minimize the perpendicular distance
    _, _, V = np.linalg.svd(A)
    
    # The last row of V gives the parameters A, B, and C of the line Ax + By + C = 0
    coeffs = V[-1]
    A, B, C = coeffs
    
    # Slope and intercept from A, B, and C
    m = -A / B
    c = -C / B
    
    # Calculate perpendicular distances for each point
    distances = np.abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)
    
    # RMS of perpendicular deviations
    rms_perpendicular = np.sqrt(np.mean(distances ** 2))

    # Calculate predicted y values based on the homogeneous fit
    y_pred = m * x + c

    # RMS of vertical deviations (standard error in y)
    rms_vertical = np.sqrt(np.mean((y - y_pred) ** 2))
    
    return m, c, rms_vertical, rms_perpendicular

# Non-homogeneous least squares fit function
def non_homogeneous_least_squares(x, y):
    # Number of data points
    n = len(x)
    
    # Calculate the slope (m) and intercept (c)
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    c = (np.sum(y) - m * np.sum(x)) / n
    
    # Predicted y values
    y_pred = m * x + c
    
    # RMS of vertical deviations
    rms_vertical = np.sqrt(np.sum((y - y_pred) ** 2) / n)
    
    # RMS of perpendicular deviations
    rms_perpendicular = np.sqrt(np.sum((y - y_pred)**2 + (x - (y - c) / m)**2) / n)
    
    return m, c, rms_vertical, rms_perpendicular


# Perform homogeneous least squares fit
m_h, c_h, rms_vertical_h, rms_perpendicular_h = homogeneous_least_squares(line1[0], line1[1])
# Perform non-homogeneous least squares fit
m_nh, c_nh, rms_vertical_nh, rms_perpendicular_nh = non_homogeneous_least_squares(line1[0], line1[1])

results = {"Output":["Slope","Intercept","RMS Vertical Deviations","RMS Perpendicular Deviations"],
           "Homogenous": [m_h,c_h,rms_vertical_h,rms_perpendicular_h],
           "Non-Homogenous": [m_nh,c_nh,rms_vertical_nh,rms_perpendicular_nh]}
a1 = pd.DataFrame(results)
a1 = a1.round(4)
print("Line 1\n",a1,"\n")

ynh = (line1[0]*m_nh) + c_nh
yh = (line1[0]*m_h) + c_h
plt.scatter(line1[0],line1[1],c="w",ec="C0")
plt.plot(line1[0],yh,label="homogenous")
plt.plot(line1[0],ynh,label="non-homogenous")
plt.legend(loc="upper right"),plt.title("Line 1 Data")
plt.xlabel("axis 1"),plt.ylabel("axis 2")
plt.savefig(f"{save_dir}/f1_line1.png",dpi=200,bbox_inches="tight")
plt.close()

# Perform homogeneous least squares fit
m_h, c_h, rms_vertical_h, rms_perpendicular_h = homogeneous_least_squares(line2[0], line2[1])
# Perform non-homogeneous least squares fit
m_nh, c_nh, rms_vertical_nh, rms_perpendicular_nh = non_homogeneous_least_squares(line2[0], line2[1])

results = {"Output":["Slope","Intercept","RMS Vertical Deviations","RMS Perpendicular Deviations"],
           "Homogenous": [m_h,c_h,rms_vertical_h,rms_perpendicular_h],
           "Non-Homogenous": [m_nh,c_nh,rms_vertical_nh,rms_perpendicular_nh]}
a1 = pd.DataFrame(results)
a1 = a1.round(4)
print("Line 2\n",a1)

ynh = (line2[0]*m_nh) + c_nh
yh = (line2[0]*m_h) + c_h
plt.scatter(line2[0],line2[1],c="w",ec="C0")
plt.plot(line2[0],yh,label="homogenous")
plt.plot(line2[0],ynh,label="non-homogenous")
plt.legend(loc="upper right"),plt.title("Line 2 Data")
plt.xlabel("axis 1"),plt.ylabel("axis 2")
plt.savefig(f"{save_dir}/f2_line2.png",dpi=200,bbox_inches="tight")
plt.close()


print("Questions completed: ",2)