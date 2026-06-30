# NACA Airfoil OpenFOAM Automation

This repository contains a modular, automated software pipeline for simulating aerodynamic forces on NACA 4-digit airfoils using OpenFOAM. The project is designed to be fully reproducible from the command line, setting up directories, running simulations, and generating plots automatically.

## Project Structure

- `src/core_utils.py`: Contains the core logic and functions for generating NACA airfoil points, creating OpenFOAM configuration files, and interacting with the solver.
- `src/convergence_study.py`: Main script to perform a spatial convergence study by varying the number of points on a specific airfoil surface.
- `src/parameter_study.py`: Main script to perform a parameter study, evaluating the impact of the airfoil's maximum camber on lift and drag coefficients.
- `src/bonus_task.py`: A supplementary script that deliberately generates a low-resolution mesh on a highly cambered airfoil to demonstrate meshing failures.
- `SUB_SCRIPT_CONVERGENCE`: Example SLURM batch script to run the simulations in an HPC environment.
- `example_baseline.py`: The original, monolithic baseline script provided as reference.

---

## Installation and Environment Setup

To ensure all scripts run correctly and do not conflict with your system Python packages, it is highly recommended to set up a virtual environment and install the required dependencies using the provided `requirements.txt`.

1. **Clone the repository and navigate into it:**
   ```bash
   git clone https://github.com/Stokes7/NACA_Airfoil_OpenFoam.git
   cd NACA_Airfoil_OpenFoam
   ```

2. **Create a Python virtual environment:**
   ```bash
   python3 -m venv .env
   ```

3. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source .env/bin/activate
     ```
   - On Windows:
     ```bash
     .env\Scripts\activate
     ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 1. Convergence Study (`src/convergence_study.py`)

This script evaluates how the mesh density (number of points defining the airfoil surface) affects the accuracy of the aerodynamic forces and the stability of the solver.

### What it generates:
- A root folder named `convergence_study_num_points/`.
- Individual OpenFOAM case directories for each point count evaluated (e.g., `openfoam_naca_100`, `openfoam_naca_200`).
- Plots analyzing spatial convergence, relative errors, residual history, and temporal stability (saved automatically in the `figs/` directory).

### What is hardcoded (can be modified by the user):
- `airfoil_code = "8412"`: The specific NACA profile used for the convergence test.
- `chord_length = 1.0`: The chord of the airfoil in meters.
- `n_points = list(range(1000, 2050, 50))`: The list or range of point counts to evaluate.
- Simulation free-stream variables inside the `run_simulation` function (`rho_freestream = 1.225`, `u_freestream = 30`, `nu = 1.5e-5`).

**To run:**
```bash
python3 src/convergence_study.py
```

---

## 2. Parameter Study (`src/parameter_study.py`)

This script evaluates the aerodynamic impact of the maximum camber (the first digit of the NACA 4-series code). 

### What it generates:
- A root folder named `parameter_study_camber/`.
- Individual OpenFOAM cases for each NACA variation evaluated.
- A final plot contrasting $C_l$ and $C_d$ against the camber percentage (`figs/camber_parameter_study.png`).

### What is hardcoded:
- `cambers = list(range(0, 9))`: The range of camber values to evaluate (from NACA 0412 up to 8412).
- `n_surface_points = 600`: The optimal number of points found during the convergence study.
- `chord_length = 1.0`
- `end_time = 0.5`: The duration in seconds of the steady-state simulation.

**To run:**
```bash
python3 src/parameter_study.py
```

---

## 3. Core Utilities (`src/core_utils.py`)

This module is the backbone of the repository. It adheres to DRY (Don't Repeat Yourself) principles by housing all the complex geometry math and OpenFOAM file generation logic so the main scripts remain clean.

**Key functions included:**
- `generate_naca4()`: Computes the actual physical (x, y) coordinates of the airfoil based on the mathematical equations for the NACA 4-digit series.
- `curiosityFluidsAirfoilMesher()`: A meshing algorithm that takes the surface points and generates a structured, multi-block `blockMeshDict` for OpenFOAM.
- `setup_case_directories()`: Creates the standard OpenFOAM folder structure (`0`, `constant`, `system`) for a fresh case.
- `generate_0_directory()`: Sets up the initial and boundary condition files for Velocity (`U`), Pressure (`p`), and Turbulence parameters (`nuTilda`).
- `generate_physical_properties()` & `create_momentum_transport_sa()`: Configures the kinematic viscosity and activates the Spalart-Allmaras turbulence model.
- `create_control_dict()`: Configures the solver run settings, time steps, and crucially, sets up the `forceCoeffs` function object to extract Lift and Drag seamlessly during the run.
- `foam_cmd()`: A simple helper to safely format commands like `blockMesh` or `foamRun` for Python's `subprocess`.
