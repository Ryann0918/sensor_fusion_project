# Sensor Fusion Project

A collection of sensor fusion tasks and algorithms (e.g., calibration, EKF, particle filter) for coursework/experiments.

## Project Structure

- `code/` — Source scripts for different tasks
- `data/` — Datasets and inputs (ignored by Git)
- `output/` — Results and generated artifacts (ignored by Git)
- `.history/` — Local history (ignored by Git)

## Available Scripts (examples)

- `code/camera_calibration_task3.py`
- `code/part1_solution.py`
- `code/task5_solution.py`
- `code/task6_solution.py`
- `code/task67_solution.py`
- `code/task7_ekf_optimal.py`
- `code/task7_particle_filter.py`
- `code/task7_particle_filter_fixed.py`

## Getting Started

1. Create a virtual environment (optional but recommended):
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
2. Install the Python packages you need based on each script's imports.
3. Run any script, for example:
   ```powershell
   python code\task7_ekf_optimal.py
   ```

## Data & Outputs

- Place input datasets under `data/`.
- Generated results are written to `output/`.
- Both folders are ignored by Git by default. If you need small sample files tracked, add exceptions or commit examples in subfolders.

## Notes

- Python 3.9+ recommended. Adjust as needed for your environment.
- Feel free to add `requirements.txt` once dependencies are finalized.

---
Maintained by: @Ryann0918
