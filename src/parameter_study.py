import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from core_utils import (
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
def run_simulation(case_name, airfoil_code="8412", chord=1.0, points=200, end_time=0.6):

    # OpenFOAM parameters
    rho_freestream = 1.225
    u_freestream = 30
    nu = 1.5e-5

    # Basic directory setup
    setup_case_directories(case_name)

    # Generate points
    discretized_surface_points = generate_naca4(airfoil_code, chord=chord, points=points)

    # Create mesh config files 
    curiosityFluidsAirfoilMesher(discretized_surface_points, f"{case_name}/system/blockMeshDict")
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

# %% ── Main Parameter Study ───────────────────────────────────────────────────
if __name__ == "__main__":
    chord_length = 1.0  # 1 meter long
    n_surface_points = 600 # Fixed optimal number of points from convergence study
    end_time = 0.55 # Fixed optimal time value from convergence study
    
    cambers = list(range(0, 9))  # M values from 0 to 8
    naca_codes = [f"{m}412" for m in cambers]
    
    cl_results = []
    cd_results = []
    
    print("--- Starting Parameter Study on Camber ---")
    base_dir = "parameter_study_camber"
    os.makedirs(base_dir, exist_ok=True)
    
    for m, code in zip(cambers, naca_codes):
        print(f"Running for NACA {code}...")
        current_case = f"{base_dir}/naca_{code}"
        
        run_simulation(current_case, airfoil_code=code, chord=chord_length, points=n_surface_points, end_time=0.5)
        cl_val, cd_val = extract_forces(current_case)
        
        cl_results.append(cl_val)
        cd_results.append(cd_val)
        
        print(f"Result for NACA {code} -> Cl: {cl_val}, Cd: {cd_val}")

    # Plot results
    if cl_results and cd_results:
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
