# Funcora

A mathematical graphing and analysis application built with Python.

Highlights
- Interactive function plotting and analysis
- Uses NumPy, Matplotlib and PySide6 for GUI
- Suitable for visualizing and exploring mathematical functions

Quickstart
1. Clone the repo
   ```bash
   git clone https://github.com/knorbay/Funcora.git
   cd Funcora
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # macOS / Linux
   source venv/bin/activate
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app
   ```bash
   # Adjust entry point if different (e.g. main.py, app.py)
   python Funcora.py
   ```

Notes
- If there is no `requirements.txt`, create one listing numpy, matplotlib, pyside6, sympy as needed:
  ```bash
  pip install numpy matplotlib pyside6 sympy
  pip freeze > requirements.txt
  ```

Contributing
- Bug reports and feature requests: open an Issue.
- Suggestions welcome — create a branch, add changes and open a Pull Request.
- Helpful labels: `good first issue`, `help wanted`.

License
This project is licensed under the MIT License — see the LICENSE file for details.

Project links
- Repository: https://github.com/knorbay/Funcora
