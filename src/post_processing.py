# %% ── Import libraries  ──────────────────────────────────────────────
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% ── Finest Mesh Force History  ──────────────────────────────────────────────
case_name = "convergence_study_num_points/openfoam_naca_1000"
forcecoeffs_file = f"{case_name}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"

if os.path.exists(forcecoeffs_file):
    df = pd.read_csv(forcecoeffs_file, sep=r'\s+', comment='#', header=None)
    time = df[0]
    cd = df[2]
    cl = df[3]

    plt.figure(figsize=(10, 5))
    plt.plot(time, cd, label="Drag Coefficient (Cd)", color="red", linewidth=2)
    plt.plot(time, cl, label="Lift Coefficient (Cl)", color="blue", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("Coefficient Value")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/force_coefficients_history_1000.png")
    # plt.show()

# %% ── Finest Mesh Residuals History  ──────────────────────────────────────────────
residuals_file = f"{case_name}/postProcessing/residuals/0/residuals.dat"

if os.path.exists(residuals_file):
    df = pd.read_csv(residuals_file, sep=r'\s+', comment='#', header=None)

    plt.figure(figsize=(10, 6))
    time = df[0]
    r_ux = df[1]
    r_uy = df[2]
    r_p = df[3]

    plt.plot(time, r_p, label='Pressure', color='black', alpha=0.8)
    plt.plot(time, r_ux, label='Velocity Ux', color='blue', alpha=0.8)
    plt.plot(time, r_uy, label='Velocity Uy', color='green', alpha=0.8)

    plt.yscale('log')
    plt.xlabel("Time (s)")
    plt.ylabel("Residual (Log Scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/residuals_history_1000.png")
    # plt.show()

# %% ── Read Convergence Study Data  ──────────────────────────────────────────────
directories = glob.glob("convergence_study_num_points/openfoam_naca_*")
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
optimal_case = "convergence_study_num_points/openfoam_naca_600"
force_file_600 = f"{optimal_case}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"

if os.path.exists(force_file_600):
    df_opt = pd.read_csv(force_file_600, sep=r'\s+', comment='#', header=None)
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

# %% ── Parameter Study Post-Processing  ──────────────────────────────────────────────
def plot_parameter_study():
    """
    Reads the data from the parameter_study_camber/ directories
    and generates the Camber vs Aerodynamic Coefficients plot.
    """
    import glob
    
    directories = glob.glob("parameter_study_camber/naca_*")
    cambers = []
    cd_results = []
    cl_results = []
    
    for d in directories:
        try:
            # Extract the first digit of the NACA 4-series (e.g. from 'naca_8412' get '8')
            code = d.split('_')[-1]
            m = int(code[0])
        except ValueError:
            continue
            
        force_file = f"{d}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"
        if os.path.exists(force_file):
            df = pd.read_csv(force_file, sep=r'\s+', comment='#', header=None)
            if not df.empty:
                # Average the last 10 steps to account for minor oscillations
                cd = df[2].iloc[-10:].mean()
                cl = df[3].iloc[-10:].mean()
                
                cambers.append(m)
                cd_results.append(cd)
                cl_results.append(cl)
                
    if cambers:
        cambers, cd_results, cl_results = zip(*sorted(zip(cambers, cd_results, cl_results)))
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Maximum Camber (1st Digit of NACA 4-series)')
        ax1.set_ylabel('Drag Coefficient (Cd)', color=color)
        ax1.plot(cambers, cd_results, marker='o', color=color, label='Cd')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.7)
        
        ax2 = ax1.twinx()  
        
        color = 'tab:blue'
        ax2.set_ylabel('Lift Coefficient (Cl)', color=color)
        ax2.plot(cambers, cl_results, marker='s', color=color, label='Cl')
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        os.makedirs("figs", exist_ok=True)
        plt.savefig("figs/camber_parameter_study.png")
        print("Saved parameter study plot to figs/camber_parameter_study.png")
        # plt.show()

plot_parameter_study()
