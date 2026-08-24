# Funcora

Funcora is a desktop mathematical graphing and analysis application developed with Python. It provides tools for visualizing mathematical functions and performing symbolic and numerical analysis through a graphical user interface.

## Features

* Function graphing
* Multiple function visualization
* Customizable X and Y axis ranges
* Function root analysis
* First and second derivative analysis
* Maximum and minimum point analysis
* Definite and indefinite integral analysis
* Mathematical expression parsing
* Graph saving
* Interactive graphical user interface
* Inline expression validation
* Toggleable grid and keyboard shortcuts
* Saveable `.funcora` project files
* Built-in Cartesian, parametric, polar, and implicit examples
* Function duplication and copyable analysis results

## Technologies

* Python
* PySide6
* SymPy
* NumPy
* Matplotlib
* Nuitka

## Project Structure

The application is built using a modular architecture that separates the graphical interface, mathematical processing, graph rendering, and analysis functionality.

## Installation

Clone the repository:

```bash
git clone https://github.com/knorbay/Funcora.git
cd Funcora
```

Install the required dependencies:

```bash
pip install PySide6 SymPy NumPy Matplotlib
```

Run the application:

```bash
python Funcora.py
```

Useful shortcuts: `Ctrl+N` adds a function, `Ctrl+S` saves the project,
`Ctrl+O` opens a project, `Ctrl+E` exports the graph, `Ctrl+0` resets the
view, and `Ctrl+Shift+G` toggles the grid.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Build for macOS

```bash
pip install pyinstaller
pyinstaller --noconfirm Funcora.spec
```

The packaged application is created at `dist/Funcora.app`.

## Download

Pre-built versions of Funcora are available through itch.io and Gumroad.

* [Download Funcora from itch.io](https://knorbay.itch.io/funcora)
* [Download Funcora from Gumroad](https://knorbay.gumroad.com/l/funcora?_gl=1*1q6wxe*_ga*MTk3NzE5ODU4LjE3ODYyMTc4NjY.*_ga_6LJN6D94N6*czE3ODYyODEzNzEkbzMkZzEkdDE3ODYyODIxODAkajQ3JGwwJGgw)

## Development

Funcora was developed as an independent software project using Python and its scientific computing ecosystem. AI-assisted development tools were also used throughout the development process for research, debugging, code analysis, and implementation.

## License

This project is licensed under the MIT License.
