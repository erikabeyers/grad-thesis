#this one has weighted sum and time periods :)

from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

model = AbstractModel()

data = DataPortal()
data.load(filename='thesis_OA12.dat')

# Indices
model.I = RangeSet(2)  # Product indices
model.J = RangeSet(3)  # Supplier indices
model.T = RangeSet(12)  # Time periods (months)

# Parameters
model.V = Param(model.I, model.J)  # Capacity of jth supplier for ith product
model.P = Param(model.I,model.J) #purchasing price of product i delivered by supplier j
model.d = Param(model.I,model.T) #total demand of product i
model.Ti = Param(model.I,model.J) #on-time delivery rate of product i offered by supplier j
model.ti = Param(model.I) #manufacturers minimum acceptable on-time delivery rate of product i
model.mu = Param(model.I,model.J) #defective rate of product i delivered by supplier j 
model.muu = Param(model.I) #manufacturers maximum acceptable defective rate of product i 
model.o = Param(model.J) #fixed ordering cost for supplier j
model.oo = Param(model.J) #variable ordering cost for supplier j 
model.tc = Param(model.J) #transportation cost of supplier j per vehicle 
model.v = Param(model.J) #vehicle capacity for supplier j in KG
model.psi = Param(model.I) #weight occupied by each unit of product i in KG
model.s = Param(model.I) #space occupied by each unit of product i in m3
model.S = 1200 #manufacturers total storage capacity in m3
model.h = Param(model.I) #holding cost ratio of product i 
model.sp = Param(model.J) #sustainability performance value of supplier j 

# Variables
model.X = Var(model.I, model.J, model.T, domain=NonNegativeReals)  # Allocated percentage of product i to supplier j in month t
model.Y = Var(model.J, model.T, domain=Binary) #Y is 1 for an order allocated to supplier j, otherwise 0 for all j

# Variable to represent the number of vehicles for transportation
model.no = Var(model.J, model.T, domain=NonNegativeReals)

#Both constraints defined as variables
model.Z1 = Var(domain=NonNegativeReals, initialize=1)
model.Z2 = Var(domain=NonNegativeReals, initialize=1)

# Constraints

# Constraint to calculate the number of vehicles based on the allocation for each month
def calculate_howmanytrucks(model, j, t):
    return model.no[j, t] == sum(model.psi[i] * model.X[i, j, t] * model.d[i, t] for i in model.I) / model.v[j]

model.howmanytrucks = Constraint(model.J, model.T, rule=calculate_howmanytrucks)

# Ensuring demand is satisfied for each product in each month
def demand_constraint_rule(model, i, t):
    return sum(model.X[i, j, t]* model.d[i, t] for j in model.J) == model.d[i, t]

model.demand_constraint = Constraint(model.I, model.T, rule=demand_constraint_rule)

# Ensuring max capacity isn't exceeded
def supplier_maxcapacity(model, i, j, t):
    return model.X[i, j, t] * model.d[i, t] <= model.V[i, j]

model.supplier_capacity_constraint = Constraint(model.I, model.J, model.T, rule=supplier_maxcapacity)

# Quality constraint ensuring standards are met for each product in each month
def quality_rule(model, i, t):
    return sum(model.mu[i, j] * model.X[i, j, t] * model.d[i, t] for j in model.J) <= model.muu[i] * model.d[i, t]

model.quality_constraint = Constraint(model.I, model.T, rule=quality_rule)

# Delivery Constraint

def delivery_constraint_rule(model, i, t):
    return sum((1-model.Ti[i, j]) * model.X[i, j, t] * model.d[i, t] for j in model.J) <= (1 - model.ti[i]) * model.d[i, t]

model.delivery_constraint = Constraint(model.I, model.T, rule=delivery_constraint_rule)

# Storage Capacity Constraint ensuring storage limits are not exceeded each month
def storage_capacity_constraint_rule(model, t):
    return sum(model.s[i] * model.X[i, j, t] * model.d[i, t] for i in model.I for j in model.J) <= model.S

model.storage_capacity_constraint = Constraint(model.T, rule=storage_capacity_constraint_rule)

# Min Cost Objective Function, adapted to include the time dimension
def calc_Z1func(model):
    purchase_costs = sum(model.P[i, j] * model.X[i, j, t] * model.d[i, t] for i in model.I for j in model.J for t in model.T)
    fixed_ordering_costs = sum(model.o[j] * model.Y[j, t] for j in model.J for t in model.T)
    variable_ordering_costs = sum(model.oo[j] * model.X[i, j, t] * model.d[i, t] for i in model.I for j in model.J for t in model.T)
    holding_costs = sum(model.h[i] * (sum(model.X[i, j, k] * model.d[i, k] for j in model.J for k in range(1, t+1))
                    - sum(model.d[i, k] for k in range(1, t+1))) for i in model.I for t in model.T)
    transportation_costs = sum(model.tc[j] * model.no[j, t] for j in model.J for t in model.T)
    return model.Z1 == purchase_costs + fixed_ordering_costs + variable_ordering_costs + holding_costs + transportation_costs

model.CZ1 = Constraint(rule=calc_Z1func)

# Max Sust. Obj as Constraint
def calc_Z2func(model):
    return model.Z2 == sum(model.sp[j] * model.X[i, j, t] * model.d[i, t] for i in model.I for j in model.J for t in model.T)

model.CZ2 = Constraint(rule=calc_Z2func)

Z1_best = 7955516.6
Z1_worst = 8656193.06
Z2_best = 387063
Z2_worst = 367823

weight_Z1 = 0.5
weight_Z2 = 0.5

model.Z1_normalized = Var()
model.Z2_normalized = Var()


def normalize_Z1(model):
    # Assuming Z1 is to be minimized, i.e., lower is better
    return model.Z1_normalized == (Z1_worst - model.Z1) / (Z1_worst - Z1_best)

def normalize_Z2(model):
    # Assuming Z2 is to be maximized, i.e., higher is better
    return model.Z2_normalized == (model.Z2 - Z2_worst) / (Z2_best - Z2_worst)

model.Normalize_Z1 = Constraint(rule=normalize_Z1)
model.Normalize_Z2 = Constraint(rule=normalize_Z2)

# Combined weighted objective
def weighted_objective_rule(model):
    # Construct a single weighted objective by combining Z1 and Z2
    return weight_Z1 * model.Z1_normalized + weight_Z2 * model.Z2_normalized

# Now, set this combined objective as the model's objective
model.WeightedObjective = Objective(rule=weighted_objective_rule, sense=maximize)

instance = model.create_instance(data)
solver = SolverFactory('glpk')
solver.solve(instance)

Z1_value = value(instance.Z1)
Z2_value = value(instance.Z2)
Z1_valuenorm = value(instance.Z1_normalized)
Z2_valuenorm = value(instance.Z2_normalized)

print(f"Optimal Z1 (Cost): {Z1_value}")
print(f"Optimal Z2 (Sustainability): {Z2_value}")

# Calculate TVSP for the weighted sum method
TVSP_weighted_sum = weight_Z1 * Z1_valuenorm + weight_Z2 * Z2_valuenorm

print(f"Normalized Z1 (Cost): {Z1_valuenorm}")
print(f"Normalized Z2 (Sustainability): {Z2_valuenorm}")
print(f"TVSP for Weighted Sum Method: {TVSP_weighted_sum}")

all_x_values = {(i,j,t): value(instance.X[i, j, t]) for i in instance.I for j in instance.J for t in instance.T}

for (i, j, t), x_val in all_x_values.items():
    # Access the demand for product i in period t
    demand_for_period = value(instance.d[i, t])
    # Multiply the allocation value by the demand
    allocated_quantity = x_val * demand_for_period
    print(f"X({i},{j},{t}) * Demand({i},{t}): {allocated_quantity} ({x_val} * {demand_for_period})")

# Generate the plot for X(i,j,t) * Demand(i,t)
time_periods = range(1, 13)  # Assuming 12 time periods
plot_data = {(i, j): [] for i in [1, 2] for j in [1, 2, 3]}

# Populate plot_data with the values of X(i,j,n) * Demand(i,n) from the model solution
for i in [1, 2]:
    for j in [1, 2, 3]:
        for t in time_periods:
            x_ijt = value(all_x_values[i, j, t])  # Replace with the appropriate Pyomo variable
            demand_it = value(instance.d[i, t])  # Replace with the appropriate Pyomo parameter
            plot_data[i, j].append(x_ijt*demand_it)

# Now create the plot
plt.figure(figsize=(10, 6))

# Plot each line
for (i, j), data in plot_data.items():
    plt.plot(time_periods, data, marker='o', label=f'Component {i}, Supplier {j}')

# Set the limits for the y-axis
plt.ylim(0, 11000)

# Add labels and title
plt.xlabel('Time Period (n)')
plt.ylabel('Allocated Quantity')
plt.title('Allocated Order Percentage Over Time for Products and Suppliers')

# Place the legend on the side of the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Make layout adjustments to make room for the legend
plt.tight_layout()

# Show the plot
plt.grid(True)
plt.show()
