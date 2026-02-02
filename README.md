# MATLAB Variable Reader for v7.3 Files

Python functions to read and list MATLAB variables from v7.3 format `.mat` files using `h5py`.

## Features

✅ **Universal reader** - Automatically detects and handles all common MATLAB data types  
✅ **Variable listing** - List all variables in a .mat file with their types and dimensions  
✅ **Timeseries objects** - Extracts time and data vectors  
✅ **Numeric arrays** - Supports 1D, 2D, 3D, and nD arrays with correct dimensions  
✅ **Strings/chars** - Decodes MATLAB character arrays  
✅ **Cell arrays** - Handles variable-length and heterogeneous cell arrays  
✅ **Structures** - Converts to Python dictionaries  
✅ **Dimension preservation** - Automatically transposes arrays to match MATLAB dimensions

## Installation

```bash
pip install h5py numpy
```

## Usage

### List all variables in a .mat file

```python
from read_matlab_variable import list_matlab_variables, print_matlab_variables

# Get dictionary of all variables
variables = list_matlab_variables("yourfile.mat")
for var_name, var_type in variables.items():
    print(f"{var_name}: {var_type}")

# Or use the pretty-print version
print_matlab_variables("yourfile.mat")
```

Output:
```
Variables in 'yourfile.mat':
--------------------------------------------------
arr1D         : double (1×1000)
arr2D         : double (50×20)
arr3D         : double (10×20×30)
file_name     : char
myStruct      : struct
tsSig         : timeseries
varLenSignals : cell

Total: 7 variable(s)
```

### Read a specific variable

```python
from read_matlab_variable import read_matlab_variable

# Read any variable from a .mat file
data = read_matlab_variable("yourfile.mat", "variable_name")
```

## Supported Data Types

| MATLAB Type | Python Return Type | Notes |
|-------------|-------------------|-------|
| **Timeseries** | `dict` with `'Time'` and `'Data'` keys | Automatically extracts time and signal data |
| **Numeric arrays** | `numpy.ndarray` | double, single, int8-64, uint8-64, logical |
| **Strings/chars** | `str` | Decoded from uint16 character arrays |
| **Cell arrays** | `list` | Can contain mixed types, variable-length arrays |
| **Structures** | `dict` | Nested structures supported |

## Examples

### Listing variables before reading

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

# 5. Read a structure
struct = read_matlab_variable("data.mat", "myStruct")
print(struct.keys()) # Field names
print(struct['field1'])  # Access field
```

### Output information

The function prints information about the variable being read:

```
Reading 'myArray':
  MATLAB class: double
  Shape: (20, 50)
  Dtype: float64
  ✓ Loaded successfully (returned shape: (50, 20))
```

## Important Notes

### Array Dimensions

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

### Timeseries Objects

For MATLAB timeseries objects, the function:
1. Identifies all timeseries variables in the file
2. Traces references through the MCOS subsystem
3. Matches time and data arrays using shape heuristics
4. Returns a dictionary with `'Time'` and `'Data'` keys

**Limitation:** If multiple timeseries have identical dimensions, the function uses index-based matching (alphabetical order of variable names).

### Cell Arrays

Cell arrays are returned as Python lists. Since cell arrays can contain heterogeneous data types and variable-length arrays, they cannot be converted to uniform NumPy arrays.

```python
cell_array = read_matlab_variable("data.mat", "myCells")
# cell_array is a list: [array1, array2, string, ...]
```

## Error Handling

If a variable is not found, the function raises a `ValueError` with a list of available variables:

```python
try:
    data = read_matlab_variable("file.mat", "nonexistent")
except ValueError as e:
    print(e)  # Variable 'nonexistent' not found. Available: ['var1', 'var2', ...]
```

## Limitations

1. **MATLAB v7.3 only** - This function is designed for v7.3 files (HDF5-based). For older formats, use `scipy.io.loadmat`.
2. **Complex timeseries** - Advanced timeseries features (events, quality info) are not extracted.
3. **Custom objects** - Custom MATLAB classes may not be fully decoded.
4. **Large files** - The entire variable is loaded into memory.

## File Format

MATLAB v7.3 files are HDF5-based. You can verify the format:

```matlab
% In MATLAB: Save as v7.3
save('data.mat', 'variable', '-v7.3');
```

## How It Works

The function:
1. Opens the `.mat` file using `h5py`
2. Checks the `MATLAB_class` attribute to identify the data type
3. Applies appropriate decoding:
   - **Timeseries**: Traces through `#subsystem#/MCOS` references
   - **Cell arrays**: Dereferences HDF5 object references
   - **Structs**: Recursively processes HDF5 groups
   - **Numeric arrays**: Transposes to restore MATLAB dimensions
   - **Strings**: Decodes uint16 character arrays

## License

Free to use and modify.

## Contributing

Feel free to extend this function for additional MATLAB types or improve the timeseries reference tracing logic.
