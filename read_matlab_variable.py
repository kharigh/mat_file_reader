import h5py
import numpy as np
import os


def list_matlab_variables(filename):
    """
    List all variables and their datatypes from a MATLAB v7.3 .mat file.
    
    Parameters:
    -----------
    filename : str
        Path to the .mat file (must be v7.3 format)
    
    Returns:
    --------
    dict
        Dictionary with variable names as keys and their MATLAB datatypes as values
        Example: {'varName1': 'double', 'varName2': 'timeseries', 'varName3': 'char'}
    
    Example:
    --------
    >>> variables = list_matlab_variables('data.mat')
    >>> for var_name, var_type in variables.items():
    >>>     print(f"{var_name}: {var_type}")
    """
    # Check if file exists
    if not os.path.exists(filename):
        abs_path = os.path.abspath(filename)
        raise FileNotFoundError(f"File not found: {filename}\nAbsolute path attempted: {abs_path}")
    
    variables = {}
    
    try:
        with h5py.File(filename, 'r') as f:
            # Iterate through all items in the root group
            for key in f.keys():
                # Skip internal HDF5 groups
                if key.startswith('#'):
                    continue
                
                item = f[key]
                
                # Determine the datatype
                if isinstance(item, h5py.Group):
                    # It's a group (likely a struct)
                    matlab_class = item.attrs.get('MATLAB_class', b'struct').decode('utf-8')
                    variables[key] = matlab_class
                elif isinstance(item, h5py.Dataset):
                    # It's a dataset
                    matlab_class = item.attrs.get('MATLAB_class', b'unknown').decode('utf-8')
                    
                    # Add shape information for arrays
                    if matlab_class in ['double', 'single', 'int8', 'int16', 'int32', 'int64', 
                                       'uint8', 'uint16', 'uint32', 'uint64', 'logical']:
                        shape = item.shape
                        if len(shape) == 1:
                            variables[key] = f"{matlab_class} (1D array, length {shape[0]})"
                        elif len(shape) == 2:
                            variables[key] = f"{matlab_class} ({shape[1]}×{shape[0]})"  # Transposed for MATLAB convention
                        else:
                            # nD array
                            shape_str = '×'.join(str(s) for s in reversed(shape))  # Reverse for MATLAB convention
                            variables[key] = f"{matlab_class} ({shape_str})"
                    else:
                        variables[key] = matlab_class
    
    except OSError as e:
        abs_path = os.path.abspath(filename)
        raise OSError(f"Unable to open file '{filename}' (tried: {abs_path}).\n"
                     f"Ensure the file is in MATLAB v7.3 format.\n"
                     f"Original error: {str(e)}")
    
    return variables


def print_matlab_variables(filename):
    """
    Print all variables and their datatypes from a MATLAB v7.3 .mat file.
    
    Parameters:
    -----------
    filename : str
        Path to the .mat file (must be v7.3 format)
    
    Example:
    --------
    >>> print_matlab_variables('data.mat')
    Variables in 'data.mat':
    -------------------------
    varName1: double (50×20)
    varName2: timeseries
    varName3: char
    """
    variables = list_matlab_variables(filename)
    
    print(f"\nVariables in '{filename}':")
    print("-" * 50)
    
    if not variables:
        print("No variables found.")
        return
    
    # Find longest variable name for alignment
    max_name_len = max(len(name) for name in variables.keys())
    
    # Print sorted by variable name
    for var_name in sorted(variables.keys()):
        var_type = variables[var_name]
        print(f"{var_name:<{max_name_len}} : {var_type}")
    
    print(f"\nTotal: {len(variables)} variable(s)")


def read_matlab_variable(filename, var_name):
    """
    Generic function to read ANY MATLAB variable from v7.3 .mat file.
    
    Handles:
    - Numeric arrays (1D, 2D, nD)
    - Strings and character arrays
    - Cell arrays
    - Timeseries objects
    - Structures
    - Other MATLAB objects
    
    Args:
        filename: Path to .mat file (v7.3 format)
        var_name: Name of variable to extract
    
    Returns:
        The variable data in appropriate Python format
    """
    
    def decode_string(ds):
        """Decode MATLAB string/char array"""
        if ds.dtype.kind == 'u':  # uint16 for chars
            return ''.join(chr(c) for c in ds[:].flatten())
        return ds[:].tobytes().decode('utf-16le', errors='ignore')
    
    def process_dataset(ds):
        """Process a dataset based on its MATLAB class"""
        attrs = dict(ds.attrs)
        matlab_class = attrs.get('MATLAB_class', b'').decode('utf-8') if 'MATLAB_class' in attrs else None
        is_empty = attrs.get('MATLAB_empty', 0)
        
        # Handle empty arrays
        if is_empty:
            return np.array([])
        
        # Handle based on MATLAB class
        if matlab_class in ['char', 'string']:
            return decode_string(ds)
        
        elif matlab_class == 'cell':
            # Cell arrays contain references to other objects
            return process_cell_array(ds)
        
        elif matlab_class in ['double', 'single', 'int8', 'int16', 'int32', 'int64',
                               'uint8', 'uint16', 'uint32', 'uint64', 'logical']:
            # Numeric arrays
            data = ds[:]
            
            # MATLAB stores arrays in column-major order (Fortran-style)
            # h5py reads them with transposed dimensions
            # We need to transpose to get original MATLAB dimensions
            
            if len(data.shape) == 1:
                # 1D array - no transpose needed
                return data
            elif data.shape.count(1) >= len(data.shape) - 1:
                # Vector (multiple dimensions but only one non-singleton) - squeeze it
                return data.squeeze()
            else:
                # Multi-dimensional array - transpose to restore MATLAB dimensions
                return data.T
        
        elif matlab_class == 'struct':
            return process_struct(ds)
        
        else:
            # Unknown type - return raw data
            return ds[:]
    
    def process_struct(ds):
        """Process MATLAB struct (contains references to actual structure)"""
        ref_type = h5py.check_dtype(ref=ds.dtype)
        if ref_type == h5py.Reference:
            with h5py.File(filename, 'r') as f:
                ref = ds[0, 0] if ds.shape[0] > 0 else None
                if ref:
                    obj = f[ref]
                    if isinstance(obj, h5py.Group):
                        return process_group(obj)
            return {}
        return ds[:]
    
    def process_cell_array(ds):
        """Process MATLAB cell array (contains references)"""
        # Check if dataset contains object references
        ref_type = h5py.check_dtype(ref=ds.dtype)
        
        if ref_type == h5py.Reference:
            # Array of references
            refs = ds[:]
            result = []
            with h5py.File(filename, 'r') as f:
                for ref in refs.flatten():
                    if ref:
                        obj = f[ref]
                        if isinstance(obj, h5py.Dataset):
                            result.append(process_dataset(obj))
                        elif isinstance(obj, h5py.Group):
                            result.append(process_group(obj))
                    else:
                        result.append(None)
            
            # Return as list (don't reshape - cell arrays can have heterogeneous shapes)
            return result
        else:
            return ds[:]
    
    def process_group(grp):
        """Process MATLAB struct or object stored as HDF5 group"""
        result = {}
        for key in grp.keys():
            item = grp[key]
            if isinstance(item, h5py.Dataset):
                result[key] = process_dataset(item)
            elif isinstance(item, h5py.Group):
                result[key] = process_group(item)
        return result
    
    def process_timeseries(f, ts_varname):
        """Special handling for timeseries objects"""
        # Find all timeseries variables
        ts_variables = []
        for key in f.keys():
            if not key.startswith('#'):
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    attrs = dict(item.attrs)
                    if attrs.get('MATLAB_class') == b'timeseries':
                        ts_variables.append(key)
        
        ts_variables.sort()
        if ts_varname not in ts_variables:
            return None
        
        ts_index = ts_variables.index(ts_varname)
        
        # Get MCOS reference array
        if '#subsystem#' not in f or 'MCOS' not in f['#subsystem#']:
            return None
        
        mcos = f['#subsystem#']['MCOS']
        
        # Find all float64 datasets in MCOS (potential timeseries data)
        all_datasets = []
        for i, ref in enumerate(mcos[0]):
            if ref:
                obj = f[ref]
                if isinstance(obj, h5py.Dataset):
                    attrs = dict(obj.attrs)
                    is_empty = attrs.get('MATLAB_empty', 0)
                    
                    if not is_empty and obj.dtype == np.float64:
                        size = np.prod(obj.shape)
                        if size > 100:
                            all_datasets.append({
                                'shape': obj.shape,
                                'data': obj[:]
                            })
        
        # Group into (time, data) pairs
        timeseries_pairs = []
        i = 0
        while i < len(all_datasets):
            time_data = None
            signal_data = None
            
            for j in range(i, min(i + 3, len(all_datasets))):
                cand = all_datasets[j]
                shape = cand['shape']
                
                if len(shape) == 2 and shape[0] == 1 and time_data is None:
                    time_data = cand
                elif len(shape) == 3 and shape[1] == 1 and shape[2] == 1 and signal_data is None:
                    signal_data = cand
                elif len(shape) == 2 and shape[1] == 1 and signal_data is None:
                    signal_data = cand
            
            if time_data and signal_data:
                timeseries_pairs.append({
                    'Time': time_data['data'].flatten(),
                    'Data': signal_data['data'].squeeze()
                })
                i = max(all_datasets.index(time_data), all_datasets.index(signal_data)) + 1
            else:
                i += 1
        
        if ts_index < len(timeseries_pairs):
            return timeseries_pairs[ts_index]
        
        return None
    
    # Main function logic
    with h5py.File(filename, 'r') as f:
        # Check if variable exists
        if var_name not in f:
            available = [k for k in f.keys() if not k.startswith('#')]
            raise ValueError(f"Variable '{var_name}' not found. Available: {available}")
        
        var = f[var_name]
        attrs = dict(var.attrs)
        matlab_class = attrs.get('MATLAB_class', b'').decode('utf-8') if 'MATLAB_class' in attrs else None
        
        print(f"Reading '{var_name}':")
        print(f"  MATLAB class: {matlab_class}")
        print(f"  Shape: {var.shape}")
        print(f"  Dtype: {var.dtype}")
        
        # Handle based on type
        if matlab_class == 'timeseries':
            result = process_timeseries(f, var_name)
            if result:
                print(f"  ✓ Timeseries with {len(result['Data'])} samples")
                return result
            else:
                print(f"  ⚠ Could not decode timeseries, returning raw data")
                return var[:]
        
        elif isinstance(var, h5py.Group):
            print(f"  Type: Group/Structure")
            return process_group(var)
        
        elif isinstance(var, h5py.Dataset):
            result = process_dataset(var)
            if isinstance(result, np.ndarray):
                print(f"  ✓ Loaded successfully (returned shape: {result.shape})")
            else:
                print(f"  ✓ Loaded successfully")
            return result
        
        else:
            print(f"  ⚠ Unknown type, returning raw data")
            return var[:]
