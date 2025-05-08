#!/usr/bin/env python
# coding: utf-8

# ## Problem 1

# In[1]:


# Cell 1: imports & raw data
import pandas as pd
import pulp as pl
import matplotlib.pyplot as plt

# Define the dataset
data = {
    "Project": ["1","2","3","4","5","6","7","8"],
    "Profit":   [2.1,0.5,3.0,2.0,1.0,1.5,0.6,1.8],
    "PersonDays":[550,400,300,350,450,500,350,200],
    "CPUTime":  [200,150,400,450,300,150,200,600]
}
df = pd.DataFrame(data).set_index("Project")
df


# In[2]:


# Cell 2: define DEA (CCR / input-oriented) solver
def dea_crs_input(df, inputs, outputs):
    """
    Returns a pd.Series of θ (efficiency scores) for each DMU in df.
    θ = 1 means on the efficient frontier.
    """
    results = {}
    for dmu in df.index:
        # create LP problem
        prob = pl.LpProblem(f"DEA_{dmu}", pl.LpMinimize)
        θ = pl.LpVariable("θ", lowBound=0)
        λ = pl.LpVariable.dicts("λ", df.index, lowBound=0)
        # objective
        prob += θ
        # input constraints: Σ λ_k * x_k <= θ * x_j
        for inp in inputs:
            prob += pl.lpSum(λ[k] * df.loc[k, inp] for k in df.index) <= θ * df.loc[dmu, inp]
        # output constraints: Σ λ_k * y_k >= y_j
        for out in outputs:
            prob += pl.lpSum(λ[k] * df.loc[k, out] for k in df.index) >= df.loc[dmu, out]
        # solve silently
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        results[dmu] = pl.value(θ)
    return pd.Series(results, name="θ")

# run it
eff = dea_crs_input(
    df,
    inputs=["PersonDays", "CPUTime"],
    outputs=["Profit"]
)


# In[3]:


# Cell 3: display efficiency table
eff_df = pd.concat([df, eff], axis=1)
eff_df["Efficient"] = eff_df["θ"].round(3) == 1.000
eff_df


# In[4]:


# Cell 4: extract reference sets (peers) & slacks for each project

# Re-solve with storing lambdas & slacks
peer_rows = []
slack_rows = []
for dmu in df.index:
    prob = pl.LpProblem(f"DEA_{dmu}", pl.LpMinimize)
    θ = pl.LpVariable("θ", lowBound=0)
    λ = pl.LpVariable.dicts("λ", df.index, lowBound=0)
    
    # objective
    prob += θ
    # inputs
    for inp in ["PersonDays","CPUTime"]:
        prob += pl.lpSum(λ[k] * df.loc[k, inp] for k in df.index) <= θ * df.loc[dmu, inp]
    # outputs
    for out in ["Profit"]:
        prob += pl.lpSum(λ[k] * df.loc[k, out] for k in df.index) >= df.loc[dmu, out]
    # solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    
    # peers (lambdas > 0)
    for k in df.index:
        val = λ[k].value()
        if val and val > 1e-6:
            peer_rows.append({"Project": dmu, "Peer": k, "Lambda": val})
    # slacks: compute residuals
    for inp in ["PersonDays","CPUTime"]:
        lhs = sum(λ[k].value() * df.loc[k, inp] for k in df.index)
        slack_rows.append({"Project": dmu, "Variable": inp, 
                           "Slack": lhs - θ.value()*df.loc[dmu, inp]})
    for out in ["Profit"]:
        lhs = sum(λ[k].value() * df.loc[k, out] for k in df.index)
        slack_rows.append({"Project": dmu, "Variable": out, 
                           "Slack": df.loc[dmu, out] - lhs})

peers_df = pd.DataFrame(peer_rows)
slacks_df = pd.DataFrame(slack_rows)
print("Reference Sets (Peers):"); display(peers_df)
print("Slacks:"); display(slacks_df)


# In[5]:


# Cell 5: bubble plot of frontier
plt.figure(figsize=(6,5))
# all points
plt.scatter(df["PersonDays"], df["CPUTime"], s=df["Profit"]*100, alpha=0.6)
for proj, row in df.iterrows():
    plt.text(row["PersonDays"]+5, row["CPUTime"]+5, proj)

# highlight efficient
for proj in eff_df[eff_df["Efficient"]].index:
    r = df.loc[proj]
    plt.scatter(r["PersonDays"], r["CPUTime"], 
                s=r["Profit"]*100, facecolors='none', edgecolors='k', linewidth=1.5)

plt.xlabel("Person-Days")
plt.ylabel("CPU Time (hrs)")
plt.title("DEA Input-Oriented CRS Frontier (bubble ∝ Profit)")
plt.grid(True)
plt.show()


# In[6]:


# Cell 6 (optional): run an input-oriented VRS model
# Simply add the constraint Σ λ_k == 1 to each LP above.

def dea_vrs_input(df, inputs, outputs):
    results = {}
    for dmu in df.index:
        prob = pl.LpProblem(f"VRS_{dmu}", pl.LpMinimize)
        θ = pl.LpVariable("θ", lowBound=0)
        λ = pl.LpVariable.dicts("λ", df.index, lowBound=0)
        prob += θ
        # input & output constraints (as before)...
        for inp in inputs:
            prob += pl.lpSum(λ[k] * df.loc[k, inp] for k in df.index) <= θ * df.loc[dmu, inp]
        for out in outputs:
            prob += pl.lpSum(λ[k] * df.loc[k, out] for k in df.index) >= df.loc[dmu, out]
        # VRS scale constraint
        prob += pl.lpSum(λ[k] for k in df.index) == 1
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        results[dmu] = pl.value(θ)
    return pd.Series(results, name="θ_VRS")

vrs_eff = dea_vrs_input(df, inputs=["PersonDays","CPUTime"], outputs=["Profit"])
pd.concat([eff_df, vrs_eff], axis=1)


# ## Problem 2

# In[7]:


pip install pulp


# In[8]:


# Cell 1: Imports & data setup
import pandas as pd
import pulp as pl

# Define factories, products, shipping costs (£/tonne), and capacities (tonnes)
factories = ["F1","F2","F3"]
products  = ["Steel","Iron"]

shipping_cost = {
    ("F1","Steel"): 200,  ("F1","Iron"):  500,
    ("F2","Steel"): 800,  ("F2","Iron"):  400,
    ("F3","Steel"): 500,  ("F3","Iron"): 1000
}

capacity = {"F1": 2000, "F2": 1500, "F3": 2500}

# Demand requirements (tonnes)
demand = {"Steel": 3200, "Iron": 1000}


# In[9]:


# Cell 2: Q1 – Minimise shipping cost with all three factories operational
model1 = pl.LpProblem("Min_Shipping_Cost", pl.LpMinimize)

# Decision vars: x[f,p] = tonnes of product p shipped from factory f
x = pl.LpVariable.dicts("x", (factories, products), lowBound=0)

# Objective: sum_{f,p} cost_{f,p} * x[f,p]
model1 += pl.lpSum(shipping_cost[(f,p)] * x[f][p]
                   for f in factories for p in products)

# Constraints:
#  1) Meet each product demand
for p in products:
    model1 += pl.lpSum(x[f][p] for f in factories) >= demand[p], f"Demand_{p}"

#  2) Do not exceed storage capacity at each factory
for f in factories:
    model1 += pl.lpSum(x[f][p] for p in products) <= capacity[f], f"Cap_{f}"

# Solve and display
model1.solve(pl.PULP_CBC_CMD(msg=False))
print("Q1 – Base case with F1, F2, F3:")
for f in factories:
    for p in products:
        print(f"  Ship {x[f][p].varValue:.0f} t of {p} from {f}")
print("  Total shipping cost = £", pl.value(model1.objective))


# In[10]:


# Cell 3: Q2 – Scenario: Factory 3 unavailable (only F1 & F2)
model2 = pl.LpProblem("No_F3", pl.LpMinimize)
# reuse x2 variables for F1 & F2 only
facs2 = ["F1","F2"]
x2 = pl.LpVariable.dicts("x2", (facs2, products), lowBound=0)

# objective
model2 += pl.lpSum(shipping_cost[(f,p)] * x2[f][p]
                   for f in facs2 for p in products)

# demands
for p in products:
    model2 += pl.lpSum(x2[f][p] for f in facs2) >= demand[p], f"Dem_{p}"

# capacities
for f in facs2:
    model2 += pl.lpSum(x2[f][p] for p in products) <= capacity[f], f"Cap_{f}"

# solve
model2.solve(pl.PULP_CBC_CMD(msg=False))
print("\nQ2 – F3 down; feasible?", pl.LpStatus[model2.status])
if pl.LpStatus[model2.status] == "Optimal":
    for f in facs2:
        for p in products:
            print(f"  Ship {x2[f][p].varValue:.0f} t of {p} from {f}")
    print("  Total cost = £", pl.value(model2.objective))


# In[11]:


# Cell 4: Q3 – Minimise combined shipping + raw material cost
# Raw material cost per tonne
raw_cost = {
    ("F1","Steel"):  50,  ("F1","Iron"):  100,
    ("F2","Steel"):  70,  ("F2","Iron"):  120,
    ("F3","Steel"):  45,  ("F3","Iron"):  130
}

model3 = pl.LpProblem("Min_Total_Cost", pl.LpMinimize)
x3 = pl.LpVariable.dicts("x3", (factories, products), lowBound=0)

# Objective: shipping + raw material
model3 += pl.lpSum(
    (shipping_cost[(f,p)] + raw_cost[(f,p)]) * x3[f][p]
    for f in factories for p in products
)

# Reapply demand & capacity constraints
for p in products:
    model3 += pl.lpSum(x3[f][p] for f in factories) >= demand[p], f"Dem_{p}"
for f in factories:
    model3 += pl.lpSum(x3[f][p] for p in products) <= capacity[f], f"Cap_{f}"

# Solve & display
model3.solve(pl.PULP_CBC_CMD(msg=False))
print("\nQ3 – Combined shipping + raw‐material cost:")
for f in factories:
    for p in products:
        print(f"  Ship {x3[f][p].varValue:.0f} t of {p} from {f}")
print("  Total combined cost = £", pl.value(model3.objective))


# VISUALIZATION:

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

unit_costs = {
    "Factory-Product": ["F1-Steel", "F1-Iron", "F2-Steel", "F2-Iron", "F3-Steel", "F3-Iron"],
    "Shipping Cost": [200, 500, 800, 400, 500, 1000],
    "Combined Cost": [250, 600, 870, 520, 545, 1130]
}
df_unit = pd.DataFrame(unit_costs)
ax = df_unit.plot(x="Factory-Product", y=["Shipping Cost","Combined Cost"], kind="bar")
ax.set_xlabel("Factory–Product")
ax.set_ylabel("Unit Cost (£/t)")
ax.set_title("Figure 1. Unit Costs: Shipping vs Shipping+Raw Material")
plt.tight_layout()
plt.show()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

used = {"F1":2000, "F2":1000, "F3":1200}
capacity = {"F1":2000, "F2":1500, "F3":2500}
cap_df = pd.DataFrame({
    "Factory": list(used.keys()),
    "Used": list(used.values()),
    "Unused": [capacity[f]-used[f] for f in used]
}).set_index("Factory")
ax2 = cap_df.plot(kind="bar", stacked=True)
ax2.set_xlabel("Factory")
ax2.set_ylabel("Tonnes")
ax2.set_title("Figure 2. Factory Capacity Utilisation (Base Plan)")
plt.tight_layout()
plt.show()


# ## PROBLEM 3

# In[16]:


pip install networkx pandas


# In[19]:


import networkx as nx
import pandas as pd

# ---------  TYPE YOUR ARC DATA HERE  ---------
edges = [
    # example rows – replace with the real ones
    # (source, target, cost_or_profit),
    (1, 2, 2),
    (1, 5, 27),
    (2, 3, 4),
    (2, 4, 10),
    (3, 4, 8),
    (4, 5, 7),
    (3, 7, 3),
    (7, 5, 9),
    (3, 11, 3),
    (11, 7, 7),
    (2, 9, 15),
    (9, 8, 12),
    (8, 7, 6),
    (8, 12, 6),
    (9, 10, 6),
    (10, 12, 15),
    # add every arc that appears in the diagram
]
if not edges:
    raise ValueError("⚠️  Fill in the real edge list first!")


# In[20]:


G_cost = nx.DiGraph()
G_cost.add_weighted_edges_from(edges, weight="cost")  # `weight` is default name

short_path   = nx.dijkstra_path(G_cost,  source=1, target=12, weight="cost")
short_length = nx.dijkstra_path_length(G_cost,      1,       12,   "cost")

print("Shortest-cost route 1 → 12:", short_path)
print("Total cost (£k):", short_length)


# In[24]:


# Longest-profit path via Bellman–Ford on negative profits
from networkx.algorithms.shortest_paths.weighted import bellman_ford_path, bellman_ford_path_length

# re-use G_profit with weight = –profit from Cell 2
try:
    long_path   = bellman_ford_path(G_profit, source, sink, weight="weight")
    long_profit = sum(G_profit[u][v]["profit"] for u,v in zip(long_path[:-1], long_path[1:]))
    print("Maximum-profit route 1 → 12:", long_path)
    print("Total expected profit (£ m):", long_profit)
except nx.NetworkXUnbounded:
    print("The graph has a positive-profit cycle reachable from 1 and 12 → profit unbounded.")


# In[25]:


import pulp as pl

# -------------- build MILP --------------
model = pl.LpProblem("Longest_Path", pl.LpMaximize)

# binary var x_uv = 1 if edge (u,v) chosen
x = {(u,v): pl.LpVariable(f"x_{u}_{v}", cat="Binary") for u,v,_ in edges}

# objective: maximise total profit
model += pl.lpSum(G_profit[u][v]["profit"] * x[(u,v)] for u,v,_ in edges)

# flow-balance constraints
for n in G_profit.nodes:
    inflow  = pl.lpSum(x[(u,v)] for u,v in x if v == n)
    outflow = pl.lpSum(x[(u,v)] for u,v in x if u == n)
    if n == source:
        model += outflow - inflow == 1
    elif n == sink:
        model += inflow - outflow == 1
    else:
        model += outflow - inflow == 0

# optional: eliminate subtours with MTZ constraints (safe here because graph small)
order = {n: pl.LpVariable(f"ord_{n}", lowBound=0, upBound=len(G_profit.nodes), cat="Integer")
         for n in G_profit.nodes}
for (u,v) in x:
    if v not in (source, sink) and u not in (source, sink):
        model += order[u] + 1 <= order[v] + (1 - x[(u,v)]) * len(G_profit.nodes)

model.solve(pl.PULP_CBC_CMD(msg=False))

# extract solution
long_path = [source]
cur = source
while cur != sink:
    next_nodes = [v for (u,v) in x if u == cur and x[(u,v)].value() == 1]
    if not next_nodes:
        raise ValueError("No path found – model infeasible?")
    cur = next_nodes[0]
    long_path.append(cur)

long_profit = sum(G_profit[u][v]["profit"] for u,v in zip(long_path[:-1], long_path[1:]))

print("Maximum-profit route 1 → 12:", long_path)
print("Total expected profit (£ m):", long_profit)


# In[ ]:




