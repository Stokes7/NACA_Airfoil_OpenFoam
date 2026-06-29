# %% ── 1. Import Libraries ──────────────────────────────────────────────
import glob
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core_utils import (
    create_control_dict,
    create_momentum_transport_sa,
    curiosityFluidsAirfoilMesher,
    foam_cmd,
    generate_0_directory,
    generate_naca4,
    generate_openfoam_fvschemes_fvsolution,
    generate_physical_properties,
    setup_case_directories,
)


# %% ── Set up simulation method  ──────────────────────────────────────────────
def run_simulation(case_name, airfoil_code="8412", chord=1.0, points=100, end_time=0.6):

    # OpenFOAM parameters
    rho_freestream = 1.225
    u_freestream = 30
    nu = 1.5e-5

    # Basic directory setup
    setup_case_directories(case_name)

    # Generate points
    discretized_surface_points = generate_naca4(airfoil_code, chord=chord, points=points)

    # Create mesh  config files 
    curiosityFluidsAirfoilMesher(discretized_surface_points,f"{case_name}/system/blockMeshDict")
    create_control_dict(case_name, rho_inf=rho_freestream, u_inf=u_freestream)

    # Write config files
    create_momentum_transport_sa(case_name)
    generate_openfoam_fvschemes_fvsolution(case_name)
    generate_physical_properties(case_name, nu_value=nu)

    # Create Initial Conditions
    generate_0_directory(case_name, u_freestream, rho_freestream)

    # Run OpenFOAM
    print(f"[{case_name}] Running blockMesh...")
    subprocess.run(foam_cmd("blockMesh", "-case", case_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(foam_cmd("checkMesh", "-case", case_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"[{case_name}] Running foamRun...")
    subprocess.run(foam_cmd("foamRun", "-case", case_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return

# %% ── Extract forces  ──────────────────────────────────────────────

def extract_forces(case_name):
    """Extracts final Cl and Cd from forceCoeffs.dat"""
    force_file = os.path.join(case_name, "postProcessing", "computeLiftDrag", "0", "forceCoeffs.dat")
    if not os.path.exists(force_file):
        return None, None
        
    df = pd.read_csv(force_file, sep=r'\s+', comment='#', header=None)
    # Average the last 10 steps to account for minor oscillations
    cd = df[2].iloc[-10:].mean()
    cl = df[3].iloc[-10:].mean()
    return cl, cd


# %% ── Set up simulation  ──────────────────────────────────────────────
airfoil_code = "8412"
chord_length = 1.0  # 1 meter long
n_surface_points = 20
base_dir = "convergence_study_num_points"
case_name = "openfoam_naca"
postprocess_columns = ["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]

n_points = list(range(2050, 3050, 50))
cl_results = []
cd_results = []

os.makedirs(base_dir, exist_ok=True)

# %% ── Initial Convergence Study  ──────────────────────────────────────────────
print("--- Starting Spatial Convergence Study ---")
for point in n_points:
    print(f"Running for {point} points...")

    current_case = f"{base_dir}/{case_name}_{point}"

    run_simulation(current_case, airfoil_code="8412", chord=1.0, points=point, end_time=0.6)
    cl_val, cd_val = extract_forces(current_case)
    
    cl_results.append(cl_val)
    cd_results.append(cd_val)

    print(f"Result for {point} points -> Cl: {cl_val}, Cd: {cd_val}")

# %% ── Read Convergence Study Data  ──────────────────────────────────────────────
directories = glob.glob(f"{base_dir}/openfoam_naca_*")
points = []
final_cd = []
final_cl = []
max_res_list = []

for d in directories:
    try:
        n_points = int(d.split('_')[-1])
    except ValueError:
        continue
        
    forcecoeffs_file = f"{d}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"
    if os.path.exists(forcecoeffs_file):
        try:
            df = pd.read_csv(forcecoeffs_file, sep=r'\s+', comment='#', header=None)
            if not df.empty:
                cd_val = df.iloc[-1, 2]
                cl_val = df.iloc[-1, 3]
                
                # Also read residuals for this case
                res_file = f"{d}/postProcessing/residuals/0/residuals.dat"
                max_res = np.nan
                if os.path.exists(res_file):
                    df_res = pd.read_csv(res_file, sep=r'\s+', comment='#', header=None)
                    if not df_res.empty:
                        max_res = max(df_res.iloc[-1, 1], df_res.iloc[-1, 2], df_res.iloc[-1, 3])
                
                points.append(n_points)
                final_cd.append(cd_val)
                final_cl.append(cl_val)
                max_res_list.append(max_res)
        except Exception as e:
            print(f"Error reading {forcecoeffs_file}: {e}")

if points:
    points, final_cd, final_cl, max_res_list = zip(*sorted(zip(points, final_cd, final_cl, max_res_list)))

# %% ── Plot Convergence vs Number of Points  ──────────────────────────────────────────────
if points:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Drag Coefficient (Cd)', color=color)
    ax1.plot(points, final_cd, marker='o', color=color, label='Cd')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.7)
    
    ax2 = ax1.twinx()  
    
    color = 'tab:blue'
    ax2.set_ylabel('Lift Coefficient (Cl)', color=color)
    ax2.plot(points, final_cl, marker='s', color=color, label='Cl')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/coefficients_convergence.png")
    # plt.show()

# %% ── Relative Error Between Successive Grids  ──────────────────────────────────────────────
if points and len(points) > 1:
    cd_array = np.array(final_cd)
    cl_array = np.array(final_cl)
    
    err_cd = np.zeros_like(cd_array)
    err_cl = np.zeros_like(cl_array)
    
    err_cd[1:] = np.abs((cd_array[1:] - cd_array[:-1]) / cd_array[1:]) * 100.0
    err_cl[1:] = np.abs((cl_array[1:] - cl_array[:-1]) / cl_array[1:]) * 100.0
    
    fig2, ax_err = plt.subplots(figsize=(10, 5))
    ax_err.plot(points[1:], err_cd[1:], marker='o', color='tab:red', label='Cd Relative Error (%)')
    ax_err.plot(points[1:], err_cl[1:], marker='s', color='tab:blue', label='Cl Relative Error (%)')
    
    ax_err.set_xlabel('Number of Points')
    ax_err.set_ylabel('Relative Error (%)')
    ax_err.set_yscale('log')
    ax_err.grid(True, which="both", linestyle="--", alpha=0.7)
    ax_err.legend()
    
    fig2.tight_layout()
    plt.savefig("figs/relative_error_convergence.png")
    # plt.show()

# %% ── Temporal Relative Error (Optimal Case)  ──────────────────────────────────────────────
optimal_points = 2200
force_file_optimal = f"convergence_study_num_points/openfoam_naca_{optimal_points}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"

if os.path.exists(force_file_optimal):
    df_opt = pd.read_csv(force_file_optimal, sep=r'\s+', comment='#', header=None)
    if not df_opt.empty:
        t_opt = df_opt[0].values
        cd_opt = df_opt[2].values
        cl_opt = df_opt[3].values
        
        cd_final = cd_opt[-1]
        cl_final = cl_opt[-1]
        
        err_cd_time = np.abs((cd_opt - cd_final) / cd_final) * 100.0
        err_cl_time = np.abs((cl_opt - cl_final) / cl_final) * 100.0
        
        epsilon = 1e-12
        err_cd_time = err_cd_time + epsilon
        err_cl_time = err_cl_time + epsilon
        
        fig4, ax_time_err = plt.subplots(figsize=(10, 5))
        ax_time_err.plot(t_opt, err_cd_time, color='tab:red', label='Cd Error vs Final Value (%)')
        ax_time_err.plot(t_opt, err_cl_time, color='tab:blue', label='Cl Error vs Final Value (%)')
        
        ax_time_err.axhline(y=1e-3, color='gray', linestyle='--', label='1e-3 Threshold')
        
        ax_time_err.set_xlabel('Time (s)')
        ax_time_err.set_ylabel('Relative Error (%)')
        ax_time_err.set_yscale('log')
        ax_time_err.grid(True, which="both", linestyle="--", alpha=0.7)
        ax_time_err.legend()
        
        fig4.tight_layout()
        plt.savefig("figs/temporal_relative_error_600.png")
        # plt.show()


# %% ── Temporal Relative Error (Optimal Case)  ──────────────────────────────────────────────
