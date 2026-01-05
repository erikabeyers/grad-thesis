#this is the gamma distribution that generates demand

import numpy as np

# Parameters
mean_demand = 25208  # Mean demand
months = 12  # Number of months for which to generate demand scenarios

# Assuming a coefficient of variation (CV) of 0.1 for demand variability
cv = 0.1
std_dev = mean_demand * cv

# Calculate the shape (k) and scale (theta) parameters for the Gamma distribution
shape = (mean_demand ** 2) / (std_dev ** 2)
scale = (std_dev ** 2) / mean_demand

# Generate demand scenarios
demand_scenarios_product1 = np.random.gamma(shape, scale, months)
demand_scenarios_product2 = np.random.gamma(shape, scale, months)

# Round to nearest whole number
rounded_demand_scenarios_product1 = np.round(demand_scenarios_product1).astype(int)
rounded_demand_scenarios_product2 = np.round(demand_scenarios_product2).astype(int)

print("Demand scenarios for Product 1:", rounded_demand_scenarios_product1)
print("Demand scenarios for Product 2:", rounded_demand_scenarios_product2)
