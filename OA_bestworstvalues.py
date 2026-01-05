#this one has time periods and calculates best and worst values to be used in 12 & 13

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
model.S = 100 #manufacturers total storage capacity in m3
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
    return sum(model.X[i, j, t]* model.d[i, t] for j in model.J) - model.d[i, t] >= 0 

model.demand_constraint = Constraint(model.I, model.T, rule=demand_constraint_rule)

#Ensuring supplier capacity isn't exceeded

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
model.OZ1 = Objective(expr= model.Z1  , sense=minimize)

# Max Sust. Obj as Constraint
def calc_Z2func(model):
    return model.Z2 == sum(model.sp[j] * model.X[i, j, t] * model.d[i, t] for i in model.I for j in model.J for t in model.T)

model.CZ2 = Constraint(rule=calc_Z2func)
model.OZ2 = Objective(expr= model.Z2, sense = maximize)

model.OZ2.deactivate()


instance = model.create_instance(data)
solver = SolverFactory('glpk')
solver.solve(instance)

results = solver.solve(instance)
# Check the results
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print('Optimal solution found')
    print('Z1 =', value(instance.Z1))
    print('Z2 =', value(instance.Z2))
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print('Model is infeasible')
else:
    print('Solver Status:', results.solver.status)
    print('Termination Condition:', results.solver.termination_condition)

#instance.pprint()

print('Z1 =', value(instance.Z1))
print('Z2 =', value(instance.Z2))
Z2_min = value(instance.Z2)

model.OZ1.deactivate()
model.OZ2.activate()

instance = model.create_instance(data)
solver = SolverFactory('glpk')
solver.solve(instance);

#instance.pprint()

print('Z1 =', value(instance.Z1))
print('Z2 =', value(instance.Z2))
Z2_max = value(instance.Z2)

model.OZ1.deactivate()
model.OZ2.activate()
