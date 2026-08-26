# Thesis Project

This repository contains a thesis-oriented research prototype for DEM-based, threat-aware mission planning of a fixed-wing aircraft. The framework evaluates classical and learning-assisted planners under common terrain, threat, aircraft, and benchmark constraints.

## Scope

The project integrates three layers:

- **Perception:** DEM terrain model, static/dynamic threat models, geometric line-of-sight risk labels, and DNN-TRE neural tactical risk estimation.
- **Decision:** Neuro-adaptive planner allocation using capability, opportunity, pressure, and fused tactical index scores.
- **Action:** RL-Pilot, K-GNP, T-GnP, feasibility checking, trajectory processing, and fixed-wing simulation.

Main entry points:

- `main.py` - batch benchmark and thesis experiment workflow.
- `gui_app.py` - desktop GUI launcher.
- `gui_runner.py` - GUI-driven benchmark execution.

## Project Structure

```text
ai_modules/      DQN pilot, DNN-TRE risk model, training/data generation scripts
aircraft/        reduced-order fixed-wing aircraft model
analysis/        benchmark and risk-map comparison utilities
environment/     DEM terrain, collision, and threat models
evaluation/      path, risk, tracking, clearance, and control metrics
planners/        A*, Dijkstra, RRT*, PSO, K-GNP, T-GnP, RL-Pilot, Neuro-Adaptive
scenarios/       benchmark scenario definitions
simulator/       fixed-wing path-following simulation loop
utils/           shared feasibility, cost, and trajectory helpers
visualization/   2D/3D plots, risk maps, animation, and benchmark figures
```

## Data and Generated Outputs

Generated benchmark figures, videos, presentations, temporary exports, and run outputs are intentionally excluded from Git through `.gitignore`.

The code expects a DEM raster. By default, `ai_modules/generate_real_dataset.py` and `main.py` try to resolve `dem_utm.tif` from the project folder or from `WCCI_DEM_PATH`.

```powershell
$env:WCCI_DEM_PATH = "C:\path\to\dem_utm.tif"
```

## Setup

Create an environment from the parent directory of this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r wcci_conference_project\requirements.txt
```

Run the benchmark as a module from the parent directory:

```powershell
python -m wcci_conference_project.main
```

Run the GUI:

```powershell
python -m wcci_conference_project.gui_app
```
The output graphs of the project: https://drive.google.com/drive/folders/17E4MzYATfJpY9WXQEgKdj-EHvqw6RPUy?usp=sharing

## Academic Positioning

The framework should be described as an implementation-aware integration of fixed-wing feasibility, tactical risk perception, and adaptive planner allocation. It does not claim theoretical global optimality. Reported planner behavior should be interpreted under the implemented DEM, threat, aircraft, and benchmark assumptions.
