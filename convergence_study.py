# %% ── 1. Import Libraries ──────────────────────────────────────────────
import subprocess

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
    

    # Basic directory setup
    setup_case_directories(case_name)

    # Generate points
    discretized_surface_points = generate_naca4(airfoil_code, chord=chord_length, points=points)

    # Create mesh
    curiosityFluidsAirfoilMesher(discretized_surface_points,f"{case_name}/system/blockMeshDict")
    create_control_dict(case_name, rho_inf=rho_freestream, u_inf=u_freestream)
    subprocess.run(foam_cmd("blockMesh", "-case", case_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(foam_cmd("checkMesh", "-case", case_name), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Write config files
    create_momentum_transport_sa(case_name)
    generate_openfoam_fvschemes_fvsolution(case_name)
    generate_physical_properties(case_name, nu_value=1.5e-5)

    # Create Initial Conditions
    generate_0_directory(case_name, u_freestream, rho_freestream)

    # Run OpenFOAM
    subprocess.run(foam_cmd("foamRun", "-case", case_name), stdout=subprocess.DEVNULL)

    return



# %% ── Read Coefficients  ──────────────────────────────────────────────
airfoil_code = "8412"
chord_length = 1.0  # 1 meter long
n_surface_points = 20
rho_freestream = 1.225
u_freestream = 30
case_name = "openfoam_naca"
postprocess_columns = ["Time", "Cm", "Cd", "Cl", "Cl(f)", "Cl(r)"]


n_points = [20]
cd = []
cl = []


for point in n_points:
    run_simulation(f"{case_name}_{point}", airfoil_code="8412", chord=1.0, points=point, end_time=0.6)

    forcecoeffs_file = f"{case_name}_{point}/postProcessing/computeLiftDrag/0/forceCoeffs.dat"
    df = pd.read_csv(forcecoeffs_file, sep=r'\s+', comment='#', header=None, names=postprocess_columns)

    cd.append(np.mean(df["Cd"].iloc[-10:])) 
    cl.append(np.mean(df["Cl"].iloc[-10:]))


# %% ── Read Coefficients  ──────────────────────────────────────────────


