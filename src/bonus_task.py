import os
import subprocess

from core_utils import (
    create_control_dict,
    curiosityFluidsAirfoilMesher,
    foam_cmd,
    generate_naca4,
    setup_case_directories,
)

def run_meshing_failure():
    # Parameters deliberately chosen to cause a mesh failure or highly defective geometry
    airfoil_code = "9412"  # Extremely cambered
    chord_length = 1.0
    n_surface_points = 10  # Extremely low point count to force discretization errors
    
    base_dir = "bonus_task_mesh_failure"
    case_name = f"{base_dir}/naca_{airfoil_code}_fail"

    os.makedirs(base_dir, exist_ok=True)
    print("--- Setting up Bonus Task: Meshing Failure Analysis ---")
    setup_case_directories(case_name)

    print(f"Generating geometry for NACA {airfoil_code} with ONLY {n_surface_points} points...")
    discretized_surface_points = generate_naca4(airfoil_code, chord=chord_length, points=n_surface_points)

    print("Creating blockMeshDict and controlDict...")
    curiosityFluidsAirfoilMesher(discretized_surface_points, f"{case_name}/system/blockMeshDict")
    create_control_dict(case_name, rho_inf=1.225, u_inf=30)

    print(f"Running blockMesh on {case_name}...")
    result = subprocess.run(
        foam_cmd("blockMesh", "-case", case_name), 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print("\nBlockMesh FAILED (as expected)! This proves the meshing algorithm breaks down.")
        print("--------------------------------------------------")
        print("\n".join(result.stderr.splitlines()[-15:]))
        print("--------------------------------------------------")
    else:
        
        print("Running checkMesh to verify mesh quality...")
        check_result = subprocess.run(
            foam_cmd("checkMesh", "-case", case_name), 
            capture_output=True, 
            text=True
        )
        if "Failed 1 or more mesh checks" in check_result.stdout or check_result.returncode != 0:
             print("CheckMesh FAILED!")
        else:
             print("CheckMesh passed")

if __name__ == "__main__":
    run_meshing_failure()
