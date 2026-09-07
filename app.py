import streamlit as st
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from branca.element import Template, MacroElement
from streamlit_folium import folium_static

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
initial_resource = st.sidebar.number_input("Initial total resource sent (kg)", value=500.0)
conv_crit = st.sidebar.radio("Convergence criteria", ("Y", "N"))
avg_consumption = st.sidebar.number_input("Avg consumption (kg/person)", value=0.2)


# ==========================================
# 2. MAIN APP LOGIC
# ==========================================
st.title("SWIRD Disaster Impact & Recovery Dashboard")
st.markdown("Simulating the dynamics of Susceptible, Waterlogged, Improving, Recovered, and Drowned populations across affected nodes in Assam.")

if st.button("Run Simulation"):
    with st.spinner("Processing multi-nodal simulation..."):
        
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
        final_S, final_W, final_I, final_R, final_D = [], [], [], [], []

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
            
            final_S.append(sol[-1, 0])
            final_W.append(sol[-1, 1])
            final_I.append(sol[-1, 2])
            final_R.append(sol[-1, 3])
            final_D.append(sol[-1, 4])
            aggregated_sol += sol

        df['Final_S'] = final_S
        df['Final_W'] = final_W
        df['Final_I'] = final_I
        df['Final_R'] = final_R
        df['Final_D'] = final_D
        df['Affected_Pop'] = df['Final_W'] + df['Final_I'] + df['Final_D']

        # Relief Centers
        df = df.sort_values(by='Affected_Pop', ascending=False).reset_index(drop=True)
        df['Is_Relief_Center'] = False
        df.loc[:min(num_relief_centers, len(df))-1, 'Is_Relief_Center'] = True

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

            legend_html = '''
            {% macro html(this, kwargs) %}
            <div style="position: fixed; bottom: 50px; left: 50px; width: 170px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                padding: 10px; border-radius: 5px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                <b>Compartments</b><br>
                <i style="background:#3186cc; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Susceptible (S)<br>
                <i style="background:#ffcc00; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Waterlogged (W)<br>
                <i style="background:#ff6600; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Improving (I)<br>
                <i style="background:#00008b; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Recovered (R)<br>
                <i style="background:#ff0000; border-radius:50%; width:12px; height:12px; display:inline-block; margin-right:5px;"></i> Drowned (D)<br>
                <i style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid red; display:inline-block; margin-right:5px;"></i> Relief Center
            </div>
            {% endmacro %}
            '''
            macro = MacroElement()
            macro._template = Template(legend_html)
            m.get_root().add_child(macro)
            
            folium_static(m, width=600, height=500)

        with col2:
            st.subheader("Time-Series Population Dynamics")
            
            total_recovered = aggregated_sol[:, 3].copy()
            total_affected = aggregated_sol[:, 1] + aggregated_sol[:, 2] + aggregated_sol[:, 4]
            sigma_base = 0.508 * base_beta - 0.112 * base_zeta
            current_resource_pool = initial_resource
            cumulative_extra_recovered = 0.0

            for i in range(1, steps):
                delta_affected = total_affected[i] - total_affected[i-1]
                t = time_array[i]
                added_resource = 0.0
                
                if conv_crit == 'Y':
                    if delta_affected > 0:
                        W_t, I_t = aggregated_sol[i-1, 1], aggregated_sol[i-1, 2]
                        denom = base_zeta * I_t - sigma_base * (1 - base_zeta) * W_t
                        if abs(denom) < 1e-8: denom = 1e-8 
                        ratio = min(abs(1.0 / denom), 1.0) 
                        added_resource = current_resource_pool * ratio
                        current_resource_pool += added_resource
                    boost = (num_relief_centers * added_resource) / max(avg_consumption, 1e-8)
                    cumulative_extra_recovered += boost

                else: 
                    if t <= 1.0:
                        gap = total_affected[i] - (aggregated_sol[i, 3] + cumulative_extra_recovered)
                        if gap > 0:
                            added_resource = (gap * max(avg_consumption, 1e-8)) / max(num_relief_centers, 1)
                        boost = (num_relief_centers * added_resource) / max(avg_consumption, 1e-8)
                        cumulative_extra_recovered += boost
                    else:
                        cumulative_extra_recovered *= 0.45 

                total_recovered[i] = aggregated_sol[i, 3] + cumulative_extra_recovered
                if total_recovered[i] > total_affected[i]:
                    total_recovered[i] = total_affected[i]
                    cumulative_extra_recovered = max(0, total_recovered[i] - aggregated_sol[i, 3])

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.fill_between(time_array, total_recovered, total_affected, color='purple', alpha=0.1)
            ax.fill_between(time_array, 0, total_recovered, color='blue', alpha=0.1)
            ax.plot(time_array, total_recovered, linewidth=2, color='blue', label='Recovered Population')
            ax.plot(time_array, total_affected, linewidth=2, color='red', label='Affected Population (W+I+D)')
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Total Population")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right')
            
            st.pyplot(fig)
