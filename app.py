import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from branca.element import Template, MacroElement
from streamlit_folium import folium_static
import pulp
import math

st.set_page_config(page_title="SWIRD Disaster Dashboard", layout="wide")

# ==========================================
# 1. SIDEBAR INPUTS
# ==========================================
st.sidebar.header("Dashboard Configuration")

num_nodes = st.sidebar.slider("Number of affected nodes (Max 90)", 1, 90, 50)
num_relief_centers = st.sidebar.number_input("Number of relief centers (N)", min_value=1, max_value=90, value=5)

st.sidebar.subheader("Initial Population (Means)")
base_S0 = st.sidebar.number_input("Susceptible (S0)", value=50000.0)
base_W0 = st.sidebar.number_input("Waterlogged (W0)", value=15000.0)
base_I0 = st.sidebar.number_input("Improving (I0)", value=10000.0)
base_R0 = st.sidebar.number_input("Recovered (R0)", value=5000.0)
base_D0 = st.sidebar.number_input("Drowned (D0)", value=2000.0)

st.sidebar.subheader("Model Rates")
base_beta = st.sidebar.slider("Rainfall chance (beta)", 0.1, 1.0, 0.6)
base_zeta = st.sidebar.slider("Drainage rate (zeta)", 0.1, 1.0, 0.3)

st.sidebar.subheader("Resource Allocation")
initial_resource = st.sidebar.number_input("Initial total resource sent (kg)", value=50000.0)
conv_crit = st.sidebar.radio("Convergence criteria", ("Y", "N"))
avg_consumption = st.sidebar.number_input("Avg consumption (kg/person)", value=0.2)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2-lon1)/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ==========================================
# 2. MAIN APP LOGIC
# ==========================================
st.title("SWIRD Disaster Impact & Integrated Resource Optimization")
st.markdown("Simulating the ODE dynamics and executing an MILP for optimal resource distribution to minimize operating and penalty costs.")

if st.button("Run Simulation & Optimization"):
    with st.spinner("Processing multi-nodal ODE simulation and solving MILP..."):
        
        # Time Constants
        T = 20
        dt = 0.01
        steps = int(T/dt)
        time_array = np.linspace(0, T, steps)

        # Load Data
        excel_file = "Assam_90_Locations_Coordinates.xlsx"
        try:
            df = pd.read_excel(excel_file, sheet_name='90 Locations Dataset', header=3)
            df = df.dropna(subset=['Latitude (°N)', 'Longitude (°E)', 'Location Name'])
        except Exception as e:
            st.error(f"Error loading data: {e}. Ensure the Excel file is uploaded to GitHub.")
            st.stop()

        available_nodes = len(df)
        if num_nodes < available_nodes:
            df = df.sample(n=num_nodes, random_state=42).reset_index(drop=True)
        else:
            num_nodes = available_nodes

        # Generate Distributions
        np.random.seed(42)
        S0_nodes = np.abs(np.random.normal(loc=base_S0, scale=base_S0*0.1, size=num_nodes))
        W0_nodes = np.abs(np.random.normal(loc=base_W0, scale=base_W0*0.2, size=num_nodes))
        I0_nodes = np.abs(np.random.normal(loc=base_I0, scale=base_I0*0.2, size=num_nodes))
        R0_nodes = np.abs(np.random.normal(loc=base_R0, scale=base_R0*0.2, size=num_nodes))
        D0_nodes = np.abs(np.random.normal(loc=base_D0, scale=base_D0*0.2, size=num_nodes))
        beta_nodes = np.clip(np.random.normal(loc=base_beta, scale=0.1, size=num_nodes), 0.1, 0.9)
        zeta_nodes = np.clip(np.random.normal(loc=base_zeta, scale=0.1, size=num_nodes), 0.1, 0.9)

        # RK4 ODE Logic
        def swird_model_stable(y, beta, zeta):
            S, W, I, R, D = y
            C = W + I
            N_pop = max(S + W + I + R + D, 1e-8)
            scale = 100.0 / N_pop 
            sigma_base = 0.508 * beta - 0.112 * zeta
            beta_eff = beta
            sigma = 0.508 * beta_eff - 0.112 * zeta
            
            dS = -beta_eff * S * W * scale - (1 - beta_eff) * S * I * scale
            dW = beta_eff * S * W * scale - sigma * W - zeta * W
            dI = (1 - beta_eff) * S * I * scale + sigma * W - zeta * (1 - sigma) * I
            dR = zeta * (1 - sigma) * I
            dD = sigma * (1 - zeta) * W
            
            return np.array([dS, dW, dI, dR, dD])

        aggregated_sol = np.zeros((steps, 5)) 
        node_trajectories = [] # Save for MILP

        for i in range(num_nodes):
            y = np.array([S0_nodes[i], W0_nodes[i], I0_nodes[i], R0_nodes[i], D0_nodes[i]])
            sol = np.zeros((steps, 5))
            for step_idx in range(steps):
                sol[step_idx] = y
                k1 = dt * swird_model_stable(y, beta_nodes[i], zeta_nodes[i])
                k2 = dt * swird_model_stable(y + 0.5*k1, beta_nodes[i], zeta_nodes[i])
                k3 = dt * swird_model_stable(y + 0.5*k2, beta_nodes[i], zeta_nodes[i])
                k4 = dt * swird_model_stable(y + k3, beta_nodes[i], zeta_nodes[i])
                y = np.clip(y + (k1 + 2*k2 + 2*k3 + k4) / 6, 0, None)
            
            node_trajectories.append(sol)
            aggregated_sol += sol

        df['Final_S'] = [sol[-1, 0] for sol in node_trajectories]
        df['Final_W'] = [sol[-1, 1] for sol in node_trajectories]
        df['Final_I'] = [sol[-1, 2] for sol in node_trajectories]
        df['Final_R'] = [sol[-1, 3] for sol in node_trajectories]
        df['Final_D'] = [sol[-1, 4] for sol in node_trajectories]
        df['Affected_Pop'] = df['Final_W'] + df['Final_I'] + df['Final_D']

        # Relief Centers Identification
        df = df.sort_values(by='Affected_Pop', ascending=False).reset_index(drop=True)
        df['Is_Relief_Center'] = False
        df.loc[:min(num_relief_centers, len(df))-1, 'Is_Relief_Center'] = True
        
        camp_indices = df[df['Is_Relief_Center']].index.tolist()
        node_indices = df.index.tolist()

        # ==========================================
        # MILP OPTIMIZATION (PuLP)
        # ==========================================
        Time = range(1, T + 1)
        Camps = range(len(camp_indices))
        Nodes = range(num_nodes)
        Comps = ['S', 'W', 'I', 'D']
        Comp_idx = {'S': 0, 'W': 1, 'I': 2, 'D': 4}
        
        # Distance Matrix 
        dist_matrix = np.zeros((len(Camps), num_nodes))
        for j, c_idx in enumerate(camp_indices):
            for i, n_idx in enumerate(node_indices):
                dist_matrix[j, i] = haversine(
                    df.loc[c_idx, 'Latitude (°N)'], df.loc[c_idx, 'Longitude (°E)'],
                    df.loc[n_idx, 'Latitude (°N)'], df.loc[n_idx, 'Longitude (°E)']
                )
                
        # MILP Parameters (Re-balanced so operation is economically viable vs penalties)
        cost_open = 5000.0      # Lowered to encourage opening
        cost_op = 1000.0        # Lowered to encourage operating
        cost_inv = 0.5          
        trans_cost_rate = 0.05  # Realistic transport cost scaling
        penalties = {'S': 50.0, 'W': 200.0, 'I': 500.0, 'D': 2000.0} # D highest, S lowest
        
        camp_capacity = initial_resource / max(len(Camps), 1)
        
        prob = pulp.LpProblem("Relief_Resource_Allocation", pulp.LpMinimize)
        
        Open = pulp.LpVariable.dicts("Open", (Camps, Time), cat='Binary')
        Operate = pulp.LpVariable.dicts("Operate", (Camps, Time), cat='Binary')
        Inv = pulp.LpVariable.dicts("Inv", (Camps, Time), lowBound=0)
        Supply = pulp.LpVariable.dicts("Supply", (Camps, Time), lowBound=0)
        Flow = pulp.LpVariable.dicts("Flow", (Camps, Nodes, Time, Comps), lowBound=0)
        Unmet = pulp.LpVariable.dicts("Unmet", (Nodes, Time, Comps), lowBound=0)
        
        prob += (
            pulp.lpSum(cost_open * Open[j][t] for j in Camps for t in Time) +
            pulp.lpSum(cost_op * Operate[j][t] for j in Camps for t in Time) +
            pulp.lpSum(cost_inv * Inv[j][t] for j in Camps for t in Time) +
            pulp.lpSum(trans_cost_rate * dist_matrix[j][i] * Flow[j][i][t][c] for j in Camps for i in Nodes for t in Time for c in Comps) +
            pulp.lpSum(penalties[c] * Unmet[i][t][c] for i in Nodes for t in Time for c in Comps)
        )
        
        for j in Camps:
            prob += pulp.lpSum(Open[j][t] for t in Time) <= 1 
            for t in Time:
                prob += Operate[j][t] <= pulp.lpSum(Open[j][tau] for tau in range(1, t + 1))
                
                flow_out = pulp.lpSum(Flow[j][i][t][c] for i in Nodes for c in Comps)
                if t == 1:
                    prob += Inv[j][t] == Supply[j][t] - flow_out
                else:
                    prob += Inv[j][t] == Inv[j][t-1] + Supply[j][t] - flow_out
                    
                prob += Inv[j][t] <= camp_capacity * Operate[j][t]
                prob += Supply[j][t] <= camp_capacity * Operate[j][t]
                
        for t in Time:
            prob += pulp.lpSum(Supply[j][t] for j in Camps) <= initial_resource
            
            step = int(t / dt) - 1
            for i, n_idx in enumerate(node_indices):
                for c in Comps:
                    pop_val = node_trajectories[n_idx][step, Comp_idx[c]]
                    demand = pop_val * avg_consumption
                    prob += pulp.lpSum(Flow[j][i][t][c] for j in Camps) + Unmet[i][t][c] == demand
                    
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        t_arr_milp = list(Time)
        op_costs_history = []
        pen_costs_history = []
        
        for t in Time:
            op_cost = sum(cost_open * Open[j][t].varValue + cost_op * Operate[j][t].varValue + cost_inv * Inv[j][t].varValue for j in Camps)
            pen_cost = sum(penalties[c] * Unmet[i][t][c].varValue for i in Nodes for c in Comps)
            op_costs_history.append(op_cost)
            pen_costs_history.append(pen_cost)


        # ==========================================
        # 3. UI LAYOUT & VISUALIZATIONS
        # ==========================================
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Geographic Distribution")
            m = folium.Map(location=[26.2006, 92.9376], zoom_start=7, tiles='OpenStreetMap')
            colors = {'S': '#3186cc', 'W': '#ffcc00', 'I': '#ff6600', 'R': '#00008b', 'D': '#ff0000'}
            scale = 0.00015 

            for idx, row in df.iterrows():
                lat, lon, loc = row['Latitude (°N)'], row['Longitude (°E)'], row['Location Name']
                pops = {'S': row['Final_S'], 'W': row['Final_W'], 'I': row['Final_I'], 'R': row['Final_R'], 'D': row['Final_D']}
                for comp, val in sorted(pops.items(), key=lambda item: item[1], reverse=True):
                    if val > 0 and not np.isnan(val):
                        folium.CircleMarker(
                            location=[lat, lon], radius=val * scale, color=colors[comp],
                            fill=True, fill_color=colors[comp], fill_opacity=0.35, weight=1,
                            popup=f"<b>{loc}</b><br>{comp}: {val:,.0f}"
                        ).add_to(m)
                if row['Is_Relief_Center']:
                    folium.RegularPolygonMarker(
                        location=[lat, lon], number_of_sides=3, radius=10, color='red',
                        fill=True, fill_color='red', weight=2, popup=f"<b>RELIEF CENTER</b><br>{loc}"
                    ).add_to(m)

            folium_static(m, width=600, height=450)
            
            st.markdown("""
            <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 5px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e0e0e0;">
                <div><i style="background:#3186cc; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></i>Susceptible (S)</div>
                <div><i style="background:#ffcc00; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></i>Waterlogged (W)</div>
                <div><i style="background:#ff6600; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></i>Improving (I)</div>
                <div><i style="background:#00008b; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></i>Recovered (R)</div>
                <div><i style="background:#ff0000; width:12px; height:12px; display:inline-block; border-radius:50%; margin-right:4px;"></i>Drowned (D)</div>
                <div><i style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 12px solid red; display:inline-block; margin-right:4px;"></i>Relief Center</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # ----------------------------------------
            # PLOT 1: RECOVERY DYNAMICS
            # ----------------------------------------
            st.subheader("Population Dynamics")
            total_recovered = aggregated_sol[:, 3].copy()
            total_affected = aggregated_sol[:, 1] + aggregated_sol[:, 2] + aggregated_sol[:, 4]
            sigma_base = 0.508 * base_beta - 0.112 * base_zeta
            current_resource_pool = initial_resource
            cumulative_extra_recovered = 0.0
            
            resource_pool_history = np.zeros(steps)
            resource_pool_history[0] = initial_resource

            for i in range(1, steps):
                delta_affected = total_affected[i] - total_affected[i-1]
                t = time_array[i]
                added_resource = 0.0
                
                if conv_crit == 'Y':
                    if delta_affected > 0:
                        total_pop_t = max(aggregated_sol[i-1].sum(), 1.0)
                        W_t, I_t = aggregated_sol[i-1, 1] / total_pop_t, aggregated_sol[i-1, 2] / total_pop_t
                        denom = base_zeta * I_t - sigma_base * (1 - base_zeta) * W_t
                        if abs(denom) < 1e-8: denom = 1e-8 
                        ratio = min(abs(1.0 / denom), 1.0) * dt
                        cap_factor = num_relief_centers / max(num_nodes, 1) 
                        added_resource = current_resource_pool * ratio * cap_factor
                        current_resource_pool += added_resource 
                        
                    boost = (num_relief_centers * added_resource) / max(avg_consumption, 1e-8)
                    cumulative_extra_recovered += boost

                else: 
                    if t <= 1.0:
                        gap = total_affected[i] - (aggregated_sol[i, 3] + cumulative_extra_recovered)
                        if gap > 0 and current_resource_pool > 0:
                            required_resource = (gap * max(avg_consumption, 1e-8)) / max(num_relief_centers, 1)
                            added_resource = min(required_resource, current_resource_pool)
                            current_resource_pool -= added_resource 
                            boost = (num_relief_centers * added_resource) / max(avg_consumption, 1e-8)
                            cumulative_extra_recovered += boost
                    else:
                        cumulative_extra_recovered *= 0.98 

                resource_pool_history[i] = current_resource_pool
                total_recovered[i] = aggregated_sol[i, 3] + cumulative_extra_recovered
                if total_recovered[i] > total_affected[i]:
                    total_recovered[i] = total_affected[i]
                    cumulative_extra_recovered = max(0, total_recovered[i] - aggregated_sol[i, 3])

            fig1, ax1 = plt.subplots(figsize=(8, 3.5))
            ax1.fill_between(time_array, total_recovered, total_affected, color='purple', alpha=0.1)
            ax1.fill_between(time_array, 0, total_recovered, color='blue', alpha=0.1)
            ax1.plot(time_array, total_recovered, linewidth=2, color='blue', label='Recovered Population')
            ax1.plot(time_array, total_affected, linewidth=2, color='red', label='Affected Population')
            ax1.set_xlabel("Time (t)")
            ax1.set_ylabel("Total Population")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend(loc='lower right')
            st.pyplot(fig1)

            # ----------------------------------------
            # PLOT 2: CAPACITY CONSTRAINT
            # ----------------------------------------
            st.subheader("Capacity Constraint Analysis")
            
            max_accommodated = (resource_pool_history / max(avg_consumption, 1e-8)) * num_relief_centers
            
            fig2, ax2 = plt.subplots(figsize=(8, 3.5))
            ax2.plot(time_array, max_accommodated, linewidth=2, color='green', linestyle='--', label='Max Accommodated Capacity')
            ax2.plot(time_array, total_affected, linewidth=2, color='red', label='Affected Population')
            ax2.set_xlabel("Time (t)")
            ax2.set_ylabel("Population / Capacity")
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend(loc='upper right')
            st.pyplot(fig2)
            
            # ----------------------------------------
            # PLOT 3: MILP OPTIMIZATION
            # ----------------------------------------
            st.subheader("MILP Cost Optimization Analysis")
            
            fig3, ax3 = plt.subplots(figsize=(8, 3.5))
            ax3.plot(t_arr_milp, op_costs_history, linewidth=2, color='green', marker='o', label='Total Operating Cost')
            ax3.plot(t_arr_milp, pen_costs_history, linewidth=2, color='red', marker='x', label='Total Penalty Cost')
            ax3.set_xlabel("Time Step (Discrete)")
            ax3.set_ylabel("Cost ($)")
            ax3.grid(True, linestyle='--', alpha=0.6)
            ax3.legend(loc='upper right')
            st.pyplot(fig3)
