# %% ── 1. Import Libraries ──────────────────────────────────────────────
import os
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

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


# %% ── Read Coefficients  ──────────────────────────────────────────────
airfoil_code = "8412"
chord_length = 1.0  # 1 meter long
n_surface_points = 20
case_name = "openfoam_naca"
postprocess_columns = ["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]


n_points = list(range(200, 1050, 50))
cl_results = []
cd_results = []

print("--- Starting Spatial Convergence Study ---")
for point in n_points:
    print(f"Running for {point} points...")
    base_dir = "convergence_study_num_points"
    os.makedirs(base_dir, exist_ok=True)
    current_case = f"{base_dir}/{case_name}_{point}"
    run_simulation(current_case, airfoil_code="8412", chord=1.0, points=point, end_time=0.6)
    cl_val, cd_val = extract_forces(current_case)
    
    cl_results.append(cl_val)
    cd_results.append(cd_val)

    print(f"Result for {point} points -> Cl: {cl_val}, Cd: {cd_val}")

