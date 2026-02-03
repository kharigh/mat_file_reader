# MATLAB v7.3 File Reader & HTML Exporter

Python tools to read, list, and export MATLAB v7.3 `.mat` files (HDF5-based) with full support for timeseries, numeric arrays, strings, cell arrays, and structures.

## Features

✅ **Universal reader** - Automatically detects and handles all common MATLAB data types  
✅ **Variable listing** - List all variables with their types and dimensions  
✅ **Timeseries objects** - Extracts time and data vectors using MCOS metadata parsing  
✅ **Numeric arrays** - Supports 1D, 2D, 3D, and nD arrays with correct dimensions  
✅ **Strings/chars** - Decodes MATLAB character arrays  
✅ **Cell arrays** - Handles variable-length and heterogeneous cell arrays  
✅ **Structures** - Converts to Python dictionaries (including nested timeseries)  
✅ **HTML export** - Generate interactive HTML reports with search, show-all, and download features  
✅ **Dimension preservation** - Automatically transposes arrays to match MATLAB dimensions

## Installation

```bash
pip install h5py numpy
```

## Modules

### `read_matlab_variable.py`

Core module for reading MATLAB v7.3 files.

### `export_matlab_to_html.py`

Exports all variables to an interactive HTML file for inspection.

---

## API Reference

### `list_matlab_variables(filename)`

List all variables and their datatypes from a MATLAB v7.3 .mat file.

**Parameters:**
- `filename` (str): Path to the .mat file (must be v7.3 format)

**Returns:**
- `dict`: Dictionary with variable names as keys and MATLAB datatypes as values

**Example:**
```python
from read_matlab_variable import list_matlab_variables

variables = list_matlab_variables("data.mat")
for var_name, var_type in variables.items():
    print(f"{var_name}: {var_type}")
```

---

### `print_matlab_variables(filename)`

Print all variables and their datatypes in a formatted table.

**Parameters:**
- `filename` (str): Path to the .mat file (must be v7.3 format)

**Example:**
```python
from read_matlab_variable import print_matlab_variables

print_matlab_variables("data.mat")
```

**Output:**
```
Variables in 'data.mat':
--------------------------------------------------
arr1D         : double (1×1000)
arr2D         : double (50×20)
myStruct      : struct
tsSig         : timeseries

Total: 4 variable(s)
```

---

### `read_matlab_variable(filename, var_name)`

Read ANY MATLAB variable from a v7.3 .mat file. Automatically detects the type and returns appropriate Python objects.

**Parameters:**
- `filename` (str): Path to the .mat file (v7.3 format)
- `var_name` (str): Name of the variable to extract

**Returns:**
- Variable data in appropriate Python format (see table below)

**Supported Types:**

| MATLAB Type | Python Return Type | Notes |
|-------------|-------------------|-------|
| **Timeseries** | `dict` with `'Time'` and `'Data'` keys | Extracts time and signal data |
| **Numeric arrays** | `numpy.ndarray` | double, single, int8-64, uint8-64, logical |
| **Strings/chars** | `str` | Decoded from uint16 character arrays |
| **Cell arrays** | `list` | Can contain mixed types |
| **Structures** | `dict` | Nested structures and timeseries supported |

**Example:**
```python
from read_matlab_variable import read_matlab_variable

# Read a timeseries
ts = read_matlab_variable("data.mat", "myTimeseries")
print(ts['Time'])    # Time vector
print(ts['Data'])    # Signal data

# Read a 2D array
arr = read_matlab_variable("data.mat", "myArray")
print(arr.shape)     # Matches MATLAB dimensions
```

---

### `export_matlab_to_html(mat_filename, output_html=None, max_display_rows=100)`

Export all MATLAB variables to an interactive HTML file with:
- Summary table with variable types and sizes
- Search box to filter variables
- Expandable data tables with "Show All" buttons
- Download CSV / Copy to clipboard features
- Special formatting for timeseries (sample count display)

**Parameters:**
- `mat_filename` (str): Path to the .mat file (must be v7.3 format)
- `output_html` (str, optional): Output HTML filename. Defaults to `<mat_filename>.html`
- `max_display_rows` (int, optional): Maximum rows to display initially (default: 100)

**Returns:**
- `str`: Path to the generated HTML file

**Example:**
```python
from export_matlab_to_html import export_matlab_to_html

# Export with default settings
export_matlab_to_html("data.mat")

# Custom output filename and row limit
export_matlab_to_html("data.mat", "report.html", max_display_rows=50)
```

**Command Line:**
```bash
python export_matlab_to_html.py data.mat
python export_matlab_to_html.py data.mat custom_output.html
```

---

## Usage Examples

### List and read variables

```python
from read_matlab_variable import list_matlab_variables, read_matlab_variable

# First, see what's available
variables = list_matlab_variables("data.mat")
print(f"Found {len(variables)} variables: {list(variables.keys())}")

# Then read what you need
if 'myArray' in variables:
    data = read_matlab_variable("data.mat", "myArray")
```

### Reading different variable types

```python
from read_matlab_variable import read_matlab_variable

# 1. Read a timeseries
ts = read_matlab_variable("data.mat", "myTimeseries")
print(ts['Time'])    # Time vector
print(ts['Data'])    # Signal data

# 2. Read a 2D array (automatically transposed to match MATLAB dimensions)
arr2d = read_matlab_variable("data.mat", "myArray")
print(arr2d.shape)   # (50, 20) if MATLAB had size(myArray) = [50, 20]

# 3. Read a string
text = read_matlab_variable("data.mat", "myString")
print(text)          # 'Hello World'

# 4. Read a cell array
cells = read_matlab_variable("data.mat", "myCellArray")
print(len(cells))    # Number of cells
print(cells[0])      # First cell content

# 5. Read a structure (may contain nested timeseries)
struct = read_matlab_variable("data.mat", "myStruct")
print(struct.keys()) # Field names
print(struct['field1'])  # Access field
```

### Generate HTML report

```python
from export_matlab_to_html import export_matlab_to_html

# Export all variables to interactive HTML
export_matlab_to_html("data.mat")
# Opens data.html in browser with searchable, downloadable tables
```

---

## Array Dimensions

MATLAB stores arrays in **column-major order** (Fortran-style), while Python/NumPy uses **row-major order**. This function automatically transposes multi-dimensional arrays to preserve the original MATLAB dimensions.

- **1D arrays**: No transpose needed
- **Vectors** (e.g., (1000, 1)): Squeezed to 1D
- **2D/3D/nD arrays**: Transposed automatically

**Example:**
```matlab
% In MATLAB: Create 50×20 array
A = rand(50, 20);
save('data.mat', 'A', '-v7.3');
```

```python
# In Python: Read with correct dimensions
A = read_matlab_variable("data.mat", "A")
print(A.shape)  # (50, 20) ✓ Matches MATLAB
```

---

## Timeseries Parsing

### How It Works

MATLAB v7.3 stores timeseries objects using the MCOS (MATLAB Class Object System) format within the HDF5 structure. The parsing algorithm:

1. **Metadata Validation**: Parses the MCOS metadata blob to confirm `Time_` and `Data_` properties exist
2. **Reference Collection**: Collects all timeseries variables (including nested in structs) with their `ref_idx` values
3. **Sorting**: Sorts timeseries by `ref_idx` (ascending order)
4. **Slot Allocation**: Maps sorted position to MCOS array indices
5. **Data Extraction**: Extracts Time and Data arrays based on allocated slots

### ⚠️ Timeseries Limitations

**Important limitations when parsing MATLAB timeseries:**

1. **Positional Mapping**: The `ref_idx` value in timeseries metadata indicates *order*, not a direct pointer to data. The algorithm relies on matching timeseries to data slots by their sorted position.

2. **Alphabetical Assumption**: MATLAB assigns `ref_idx` values in file traversal order (typically alphabetical by variable name). If timeseries were created in a different order or have unusual naming, mapping may fail.

3. **Same-Length Ambiguity**: When multiple timeseries have identical sample counts, the algorithm cannot distinguish them by shape alone - it relies entirely on positional mapping.

4. **Nested Timeseries**: Timeseries inside structs are supported but follow the same positional mapping rules.

5. **Storage Format Variations**: Different MATLAB versions may store Time/Data pairs differently:
   - Some files store Time and Data as alternating arrays (ratio 2:1)
   - Some files store only Time with shape (1, N), Data with different shape
   - The algorithm uses metadata + ratio heuristics to detect the format

6. **Not Extracted**:
   - Quality information (`Quality_` property)
   - Events attached to timeseries
   - Custom timeseries metadata
   - `TimeInfo` and `DataInfo` properties

7. **Validation**: There is no guaranteed way to verify the Time/Data pairing is correct without comparing against original MATLAB data.

**Best Practices for Reliable Parsing:**
- Use consistent, alphabetically-ordered naming for timeseries variables
- Avoid mixing timeseries with very different sample counts in the same file
- Verify extracted data against known values when possible

---

## Error Handling

If a variable is not found, the function raises a `ValueError` with a list of available variables:

```python
try:
    data = read_matlab_variable("file.mat", "nonexistent")
except ValueError as e:
    print(e)  # Variable 'nonexistent' not found. Available: ['var1', 'var2', ...]
```

---

## General Limitations

1. **MATLAB v7.3 only** - Designed for v7.3 files (HDF5-based). For older formats, use `scipy.io.loadmat`.
2. **Memory usage** - The entire variable is loaded into memory.
3. **Custom objects** - Custom MATLAB classes (other than timeseries) may not be fully decoded.
4. **String arrays** - MATLAB R2016b+ string arrays may have limited support.
5. **Sparse matrices** - Sparse matrices are not specially handled.

---

## File Format

MATLAB v7.3 files are HDF5-based. You can verify/create the format:

```matlab
% In MATLAB: Save as v7.3
save('data.mat', 'variable', '-v7.3');

% Check file format
whos -file data.mat
```

---

## Internal Functions

These functions are used internally by `read_matlab_variable`:

| Function | Purpose |
|----------|---------|
| `decode_string(ds)` | Decode MATLAB string/char arrays from uint16 |
| `process_dataset(ds)` | Route dataset processing based on MATLAB class |
| `process_struct(ds)` | Process MATLAB struct references |
| `process_cell_array(ds)` | Process cell array references |
| `process_group(grp)` | Process HDF5 groups (structs) recursively |
| `process_timeseries(f, ts_varname)` | Extract timeseries using allocation algorithm |
| `_get_timeseries_structure_from_metadata(f, mcos)` | Parse MCOS metadata blob for property names |
| `_build_timeseries_allocation(f, mcos)` | Build mapping of timeseries names to MCOS indices |

---

## License

Free to use and modify.

## Contributing

Feel free to extend for additional MATLAB types or improve the timeseries reference tracing logic.
