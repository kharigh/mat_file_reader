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
    
    def process_group(grp, parent_path=""):
        """Process MATLAB struct or object stored as HDF5 group"""
        result = {}
        grp_name = grp.name.lstrip('/')  # Get group name without leading /
        
        for key in grp.keys():
            item = grp[key]
            # Build path for nested items (e.g., "myStruct/tsSig")
            item_path = f"{grp_name}/{key}"
            
            if isinstance(item, h5py.Dataset):
                # Check if this is a timeseries
                attrs = dict(item.attrs)
                if attrs.get('MATLAB_class') == b'timeseries':
                    # Process as timeseries using the full path
                    ts_result = process_timeseries(f, item_path)
                    if ts_result is not None:
                        result[key] = ts_result
                    else:
                        result[key] = process_dataset(item)
                else:
                    result[key] = process_dataset(item)
            elif isinstance(item, h5py.Group):
                result[key] = process_group(item, item_path)
        return result
    
    def process_timeseries(f, ts_varname):
        """Special handling for timeseries objects - uses allocation-based approach.
        
        MATLAB v7.3 stores timeseries with ref_idx values that indicate ORDER,
        not the actual MCOS slot. We must use the allocation algorithm to determine
        the correct time array for each timeseries.
        """
        # Validate timeseries variable exists
        if ts_varname not in f:
            print(f"  [DEBUG] Timeseries '{ts_varname}' not found in file")
            return None
        
        ts_var = f[ts_varname]
        if not isinstance(ts_var, h5py.Dataset):
            print(f"  [DEBUG] Timeseries '{ts_varname}' is not a Dataset")
            return None
        
        # Get MCOS reference array
        if '#subsystem#' not in f or 'MCOS' not in f['#subsystem#']:
            print(f"  [DEBUG] No MCOS subsystem found in file")
            return None
        
        mcos = f['#subsystem#']['MCOS']
        mcos_refs = mcos[0][:]  # Read entire row into memory to avoid repeated HDF5 access
        
        # ALWAYS use allocation algorithm (ref_idx does NOT directly map to slots)
        allocation = _build_timeseries_allocation(f, mcos_refs)
        
        # Look up this timeseries in the allocation table
        if ts_varname not in allocation:
            print(f"  [DEBUG] Timeseries '{ts_varname}' not in allocation table")
            print(f"  [DEBUG] Available in allocation: {list(allocation.keys())}")
            # Try fallback: use direct ref_idx from timeseries metadata
            return _fallback_timeseries_extraction(f, ts_var, mcos_refs)
        
        time_idx = allocation[ts_varname]
        time_ref = mcos_refs[time_idx]
        time_obj = f[time_ref]
        
        # Validate this is actually time data (float64, reasonable shape)
        if not isinstance(time_obj, h5py.Dataset):
            print(f"  [DEBUG] time_obj at index {time_idx} is not a Dataset")
            return _fallback_timeseries_extraction(f, ts_var, mcos_refs)
        
        if time_obj.dtype != np.float64:
            print(f"  [DEBUG] time_obj dtype is {time_obj.dtype}, expected float64")
            return _fallback_timeseries_extraction(f, ts_var, mcos_refs)
        
        time_data = time_obj[:]
        
        # Sanity check: time data should be numeric and have reasonable values
        if time_data.size == 0:
            print(f"  [DEBUG] time_data is empty")
            return _fallback_timeseries_extraction(f, ts_var, mcos_refs)
        
        # Find corresponding DATA array (search forward from time array)
        signal_data = None
        n_samples = time_data.shape[1] if len(time_data.shape) == 2 else time_data.shape[0]
        
        for offset in range(1, 20):
            idx = time_idx + offset
            if idx >= len(mcos_refs) or not mcos_refs[idx]:
                continue
            
            data_ref = mcos_refs[idx]
            data_obj = f[data_ref]
            
            if not isinstance(data_obj, h5py.Dataset) or data_obj.dtype != np.float64:
                continue
            
            # Data can have various shapes: (N, ...) or (1, N) or (N, 1, 1)
            # Check if any dimension matches n_samples
            if n_samples in data_obj.shape:
                signal_data = data_obj[:]
                break
        
        if signal_data is not None:
            return {
                'Time': time_data.flatten(),
                'Data': signal_data.squeeze()
            }
        
        print(f"  [DEBUG] Could not find matching Data array for {n_samples} samples")
        return _fallback_timeseries_extraction(f, ts_var, mcos_refs)
    
    def _fallback_timeseries_extraction(f, ts_var, mcos_refs):
        """Fallback method: try to extract timeseries using direct ref_idx.
        
        This is used when the allocation algorithm fails.
        """
        try:
            # Get ref_idx directly from timeseries metadata (position [0,4])
            ts_data = ts_var[:]
            if ts_data.shape[1] >= 5:
                ref_idx = int(ts_data[0, 4])
                print(f"  [DEBUG] Fallback: trying direct ref_idx = {ref_idx}")
                
                # Search around ref_idx for time-like arrays
                for search_idx in range(max(1, ref_idx - 5), min(len(mcos_refs), ref_idx + 20)):
                    if not mcos_refs[search_idx]:
                        continue
                    
                    try:
                        obj = f[mcos_refs[search_idx]]
                        if isinstance(obj, h5py.Dataset) and obj.dtype == np.float64:
                            if len(obj.shape) == 2 and obj.shape[0] == 1 and obj.shape[1] >= 2:
                                time_data = obj[:]
                                n_samples = time_data.shape[1]
                                
                                # Look for matching data array
                                for data_offset in range(1, 10):
                                    data_idx = search_idx + data_offset
                                    if data_idx >= len(mcos_refs) or not mcos_refs[data_idx]:
                                        continue
                                    
                                    data_obj = f[mcos_refs[data_idx]]
                                    if isinstance(data_obj, h5py.Dataset) and data_obj.dtype == np.float64:
                                        if n_samples in data_obj.shape:
                                            print(f"  [DEBUG] Fallback succeeded at idx {search_idx}")
                                            return {
                                                'Time': time_data.flatten(),
                                                'Data': data_obj[:].squeeze()
                                            }
                    except:
                        pass
        except Exception as e:
            print(f"  [DEBUG] Fallback failed: {e}")
        
        return None
    
    def _get_timeseries_structure_from_metadata(f, mcos_refs):
        """Parse MCOS metadata blob to determine timeseries structure.
        
        Args:
            f: HDF5 file handle
            mcos_refs: numpy array of MCOS references (already read from mcos[0][:])
        
        Returns dict with:
          - has_time: bool (Time_ property exists)
          - has_data: bool (Data_ property exists)
          - columns_per_ts: int (expected float64 arrays per timeseries)
        """
        try:
            meta_blob = f[mcos_refs[0]][:].flatten()
            
            # Extract property names ending with underscore (data storage properties)
            props = set()
            i = 0
            while i < len(meta_blob):
                if 32 <= meta_blob[i] < 127:
                    start = i
                    while i < len(meta_blob) and 32 <= meta_blob[i] < 127:
                        i += 1
                    if i - start >= 2:
                        s = bytes(meta_blob[start:i]).decode('ascii', errors='ignore')
                        if s.endswith('_'):
                            props.add(s)
                else:
                    i += 1
            
            has_time = 'Time_' in props
            has_data = 'Data_' in props
            
            return {
                'has_time': has_time,
                'has_data': has_data,
                'columns_per_ts': (1 if has_time else 0) + (1 if has_data else 0)
            }
        except:
            # Default assumption if metadata parsing fails
            return {'has_time': True, 'has_data': True, 'columns_per_ts': 2}
    
    def _build_timeseries_allocation(f, mcos_refs):
        """Build allocation table mapping timeseries names to MCOS time array indices.
        
        Args:
            f: HDF5 file handle
            mcos_refs: numpy array of MCOS references (already read from mcos[0][:])
        
        MATLAB v7.3 timeseries mapping - CORRECTED ALGORITHM:
        
        1. Validate structure from metadata (Time_, Data_ properties)
        2. Collect all timeseries (including nested in structs) with their ref_idx
        3. Sort timeseries by ref_idx (ascending)
        4. Position in sorted list = slot number
        5. Slot indices are time array MCOS indices in ascending order
        
        This works because MATLAB assigns ref_idx values in file traversal order
        (alphabetical), and stores time data in the same order.
        """
        # Step 0: Validate metadata structure
        ts_structure = _get_timeseries_structure_from_metadata(f, mcos_refs)
        if not ts_structure['has_time']:
            # No Time_ property - unusual timeseries format
            return {}
        
        expected_columns = ts_structure['columns_per_ts']
        # Step 1: Find all timeseries in file with their ref_idx (recursively)
        ts_list = []  # (name, ref_idx)
        
        def find_timeseries_recursive(group, path_prefix=""):
            """Recursively find all timeseries in groups"""
            for key in group.keys():
                if key.startswith('#'):
                    continue
                
                item = group[key]
                item_path = f"{path_prefix}/{key}" if path_prefix else key
                
                if isinstance(item, h5py.Dataset):
                    attrs = dict(item.attrs)
                    if attrs.get('MATLAB_class') == b'timeseries':
                        try:
                            ref_idx = int(item[0, 4])
                            ts_list.append((item_path, ref_idx))
                        except:
                            pass
                
                elif isinstance(item, h5py.Group):
                    find_timeseries_recursive(item, item_path)
        
        find_timeseries_recursive(f)
        
        # Step 2: Sort by ref_idx
        ts_sorted = sorted(ts_list, key=lambda x: x[1])
        
        # Step 3: Find all TIME array MCOS indices (slot_indices) in order
        # MATLAB stores Time/Data pairs: first is Time, second is Data
        # Find all float64 arrays with shape (1, N) where N >= 2, then take every other one
        # NOTE: Skip index 0 which is the MCOS metadata blob, not actual data
        all_arrays = []
        for i in range(1, len(mcos_refs)):  # Start from 1 to skip metadata blob
            if not mcos_refs[i]:
                continue
            
            try:
                obj = f[mcos_refs[i]]
                attrs = dict(obj.attrs)
                is_empty = attrs.get('MATLAB_empty', 0)
                
                # Additional check: skip if this looks like metadata (uint8 dtype is common for metadata)
                if obj.dtype == np.uint8:
                    continue
                
                if not is_empty and isinstance(obj, h5py.Dataset) and obj.dtype == np.float64:
                    if len(obj.shape) == 2 and obj.shape[0] == 1 and obj.shape[1] >= 2:
                        all_arrays.append(i)
            except:
                pass
        
        # Determine if Data arrays have same shape as Time arrays
        # Use metadata structure + ratio to decide:
        # - If metadata says 2 columns AND ratio ≈ 2 → take every other
        # - If metadata says 2 columns BUT ratio ≈ 1 → Data has different shape, take all
        # - If metadata says 1 column → take all
        if expected_columns >= 2 and len(ts_sorted) > 0 and len(all_arrays) >= len(ts_sorted) * 1.5:
            # Data has same shape as Time - take every other (first of each pair)
            slot_indices = all_arrays[::2]
        else:
            # Data has different shape OR only Time column - all matching are Time arrays
            slot_indices = all_arrays
        
        # Step 4: Build allocation - position in sorted list = slot number
        allocation = {}
        for slot_num, (name, ref_idx) in enumerate(ts_sorted):
            if slot_num < len(slot_indices):
                allocation[name] = slot_indices[slot_num]
        
        return allocation
    
    # Main function logic
    with h5py.File(filename, 'r') as f:
        # Check if variable exists
        if var_name not in f:
            available = [k for k in f.keys() if not k.startswith('#')]
            raise ValueError(f"Variable '{var_name}' not found. Available: {available}")
        
        var = f[var_name]
        
        # Check if it's a Group or Dataset
        if isinstance(var, h5py.Group):
            attrs = dict(var.attrs)
            matlab_class = attrs.get('MATLAB_class', b'').decode('utf-8') if 'MATLAB_class' in attrs else 'struct'
            print(f"Reading '{var_name}':")
            print(f"  MATLAB class: {matlab_class}")
            print(f"  Type: Group/Structure")
        else:
            attrs = dict(var.attrs)
            matlab_class = attrs.get('MATLAB_class', b'').decode('utf-8') if 'MATLAB_class' in attrs else None
            print(f"Reading '{var_name}':")
            print(f"  MATLAB class: {matlab_class}")
            print(f"  Shape: {var.shape}")
            print(f"  Dtype: {var.dtype}")
        
        # Handle based on type
        if matlab_class == 'timeseries':
            result = process_timeseries(f, var_name)
            if result is not None:
                print(f"  [OK] Timeseries with {len(result['Data'])} samples")
                return result
            else:
                print(f"  [WARN] Could not decode timeseries, returning raw data")
                return var[:]
        
        elif isinstance(var, h5py.Group):
            print(f"  Type: Group/Structure")
            return process_group(var)
        
        elif isinstance(var, h5py.Dataset):
            result = process_dataset(var)
            if isinstance(result, np.ndarray):
                print(f"  [OK] Loaded successfully (returned shape: {result.shape})")
            else:
                print(f"  [OK] Loaded successfully")
            return result
        
        else:
            print(f"  [WARN] Unknown type, returning raw data")
            return var[:]
