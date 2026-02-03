"""
Export all MATLAB variables from a .mat file to an interactive HTML file.
Provides human-readable inspection with show all/download capabilities.
"""

import json
import numpy as np
from read_matlab_variable import list_matlab_variables, read_matlab_variable


def export_matlab_to_html(mat_filename, output_html=None, max_display_rows=100):
    """
    Export all MATLAB variables to an interactive HTML file.
    
    Parameters:
    -----------
    mat_filename : str
        Path to the .mat file (must be v7.3 format)
    output_html : str, optional
        Output HTML filename. If None, uses mat_filename with .html extension
    max_display_rows : int, optional
        Maximum rows to display initially for large arrays (default: 100)
    
    Returns:
    --------
    str
        Path to the generated HTML file
    
    Example:
    --------
    >>> export_matlab_to_html('data.mat', 'data_viewer.html')
    >>> # Open data_viewer.html in any browser
    """
    
    # Determine output filename
    if output_html is None:
        output_html = mat_filename.rsplit('.', 1)[0] + '.html'
    
    # Get list of all variables
    print(f"Reading variables from '{mat_filename}'...")
    var_list = list_matlab_variables(mat_filename)
    
    # Read all variables
    variables_data = {}
    for var_name in var_list.keys():
        print(f"  Loading '{var_name}'...")
        try:
            data = read_matlab_variable(mat_filename, var_name)
            variables_data[var_name] = data
        except Exception as e:
            print(f"    [WARN] Could not read '{var_name}': {e}")
            variables_data[var_name] = None  # Mark as unreadable
    
    # Generate HTML
    print(f"Generating HTML file...")
    html_content = generate_html(mat_filename, var_list, variables_data, max_display_rows)
    
    # Write to file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Exported to '{output_html}'")
    return output_html


def generate_html(mat_filename, var_list, variables_data, max_display_rows):
    """Generate complete HTML content"""
    
    html_parts = []
    
    # HTML Header with CSS
    html_parts.append(get_html_header(mat_filename))
    
    # Summary table
    html_parts.append(generate_summary_table(var_list, variables_data))
    
    # Search box
    html_parts.append('''
    <div class="search-container">
        <input type="text" id="searchBox" placeholder="Search variables..." onkeyup="filterVariables()">
    </div>
    ''')
    
    # Each variable section
    for var_name, var_type in sorted(var_list.items()):
        var_data = variables_data.get(var_name)
        html_parts.append(generate_variable_section(var_name, var_type, var_data, max_display_rows))
    
    # JavaScript
    html_parts.append(get_javascript())
    
    # Footer
    html_parts.append('</body></html>')
    
    return '\n'.join(html_parts)


def generate_summary_table(var_list, variables_data):
    """Generate summary table of all variables with timeseries length"""
    rows = []
    for var_name, var_type in sorted(var_list.items()):
        var_data = variables_data.get(var_name)
        
        # Determine size/length info
        size_info = ""
        if var_type == 'timeseries' and isinstance(var_data, dict):
            if 'Time' in var_data:
                length = len(var_data['Time'])
                size_info = f'<span class="ts-length">{length} samples</span>'
        elif var_type == 'struct' and isinstance(var_data, dict):
            # Check for nested timeseries in struct
            ts_info = []
            for key, val in var_data.items():
                if isinstance(val, dict) and 'Time' in val and 'Data' in val:
                    ts_info.append(f'{key}: {len(val["Time"])}')
            if ts_info:
                size_info = f'<span class="ts-length">{", ".join(ts_info)}</span>'
            else:
                size_info = f'<span class="array-shape">{len(var_data)} fields</span>'
        elif isinstance(var_data, np.ndarray):
            size_info = f'<span class="array-shape">{var_data.shape}</span>'
        elif isinstance(var_data, list):
            size_info = f'<span class="array-shape">{len(var_data)} elements</span>'
        elif isinstance(var_data, str):
            size_info = f'<span class="array-shape">{len(var_data)} chars</span>'
        
        rows.append(f'<tr><td><a href="#{var_name}">{var_name}</a></td><td>{var_type}</td><td>{size_info}</td></tr>')
    
    return f'''
    <h2>Variables Summary</h2>
    <table class="summary-table">
        <thead>
            <tr><th>Variable Name</th><th>Type</th><th>Size / Length</th></tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    '''


def generate_variable_section(var_name, var_type, var_data, max_rows):
    """Generate HTML section for a single variable"""
    
    section_id = var_name.replace(' ', '_')
    
    html = f'''
    <div class="variable-section" id="{section_id}" data-varname="{var_name}">
        <h2>{var_name} <span class="var-type">({var_type})</span></h2>
    '''
    
    # Handle unreadable variables
    if var_data is None:
        html += '<p class="error">⚠ Could not read this variable (see console for details)</p></div>'
        return html
    
    if isinstance(var_data, str) and var_data.startswith("Error:"):
        html += f'<p class="error">{var_data}</p></div>'
        return html
    
    # Handle different data types
    if isinstance(var_data, np.ndarray):
        html += generate_array_html(var_name, var_data, max_rows)
    elif isinstance(var_data, dict):
        # Could be timeseries or struct
        if 'Time' in var_data and 'Data' in var_data:
            html += generate_timeseries_html(var_name, var_data, max_rows)
        else:
            html += generate_struct_html(var_name, var_data, max_rows)
    elif isinstance(var_data, list):
        # Cell array
        html += generate_cell_array_html(var_name, var_data, max_rows)
    elif isinstance(var_data, str):
        html += f'<p class="string-value">"{var_data}"</p>'
    else:
        html += f'<pre>{str(var_data)}</pre>'
    
    html += '</div>'
    return html


def generate_array_html(var_name, arr, max_rows):
    """Generate HTML for numeric arrays"""
    
    html = []
    
    # Summary stats
    html.append('<div class="stats">')
    html.append(f'<strong>Shape:</strong> {arr.shape} | ')
    html.append(f'<strong>Dtype:</strong> {arr.dtype} | ')
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        html.append(f'<strong>Min:</strong> {np.min(arr):.4g} | ')
        html.append(f'<strong>Max:</strong> {np.max(arr):.4g} | ')
        html.append(f'<strong>Mean:</strong> {np.mean(arr):.4g}')
    html.append('</div>')
    
    # Handle based on dimensions
    if arr.ndim == 0:
        # Scalar
        html.append(f'<p class="scalar-value"><strong>Value:</strong> {arr.item()}</p>')
    elif arr.size == 0:
        # Empty array
        html.append(f'<p class="empty-array"><em>Empty array</em></p>')
    elif arr.ndim == 1:
        html.append(generate_1d_array_table(var_name, arr, max_rows))
    elif arr.ndim == 2:
        html.append(generate_2d_array_table(var_name, arr, max_rows))
    else:
        # 3D+ arrays: show first slice
        html.append(f'<p><em>Showing first slice of {arr.ndim}D array (shape: {arr.shape})</em></p>')
        first_slice = arr.reshape(arr.shape[0], -1)
        html.append(generate_2d_array_table(var_name, first_slice, max_rows))
    
    return ''.join(html)


def generate_1d_array_table(var_name, arr, max_rows):
    """Generate HTML table for 1D array"""
    
    total_rows = len(arr)
    show_all = total_rows <= max_rows
    display_rows = total_rows if show_all else max_rows
    
    # Create full data as JSON for JavaScript
    full_data_json = json.dumps(arr.tolist())
    
    html = []
    html.append(f'<div class="array-container">')
    
    # Table
    html.append(f'<table class="data-table" id="table_{var_name}">')
    html.append('<thead><tr><th>Index</th><th>Value</th></tr></thead>')
    html.append('<tbody>')
    
    for i in range(display_rows):
        html.append(f'<tr><td>{i}</td><td>{arr[i]}</td></tr>')
    
    html.append('</tbody></table>')
    
    # Hidden data
    html.append(f'<script type="application/json" id="data_{var_name}">{full_data_json}</script>')
    
    # Buttons if truncated
    if not show_all:
        hidden_count = total_rows - display_rows
        html.append(f'<p class="truncated-msg">... {hidden_count} more rows hidden ...</p>')
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="showAllRows(\'{var_name}\', 1)">Show All ({total_rows} rows)</button>')
        html.append(f'<button onclick="downloadCSV(\'{var_name}\', 1)">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', 1)">Copy Data</button>')
        html.append(f'</div>')
    else:
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="downloadCSV(\'{var_name}\', 1)">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', 1)">Copy Data</button>')
        html.append(f'</div>')
    
    html.append('</div>')
    return ''.join(html)


def generate_2d_array_table(var_name, arr, max_rows):
    """Generate HTML table for 2D array"""
    
    total_rows, total_cols = arr.shape
    show_all = total_rows <= max_rows
    display_rows = total_rows if show_all else max_rows
    
    # Limit columns for very wide arrays
    max_display_cols = 20
    display_cols = min(total_cols, max_display_cols)
    cols_truncated = total_cols > max_display_cols
    
    # Create full data as JSON
    full_data_json = json.dumps(arr.tolist())
    
    html = []
    html.append(f'<div class="array-container">')
    
    # Table
    html.append(f'<table class="data-table" id="table_{var_name}">')
    html.append('<thead><tr><th>Row</th>')
    for j in range(display_cols):
        html.append(f'<th>{j}</th>')
    if cols_truncated:
        html.append(f'<th>...</th>')
    html.append('</tr></thead>')
    html.append('<tbody>')
    
    for i in range(display_rows):
        html.append(f'<tr><td>{i}</td>')
        for j in range(display_cols):
            html.append(f'<td>{arr[i, j]:.6g}</td>')
        if cols_truncated:
            html.append(f'<td>...</td>')
        html.append('</tr>')
    
    html.append('</tbody></table>')
    
    # Hidden data
    html.append(f'<script type="application/json" id="data_{var_name}">{full_data_json}</script>')
    
    # Messages and buttons
    if cols_truncated:
        html.append(f'<p class="truncated-msg">Showing {display_cols} of {total_cols} columns</p>')
    
    if not show_all:
        hidden_count = total_rows - display_rows
        html.append(f'<p class="truncated-msg">... {hidden_count} more rows hidden ...</p>')
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="showAllRows(\'{var_name}\', 2)">Show All ({total_rows}×{total_cols})</button>')
        html.append(f'<button onclick="downloadCSV(\'{var_name}\', 2)">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', 2)">Copy Data</button>')
        html.append(f'</div>')
    else:
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="downloadCSV(\'{var_name}\', 2)">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', 2)">Copy Data</button>')
        html.append(f'</div>')
    
    html.append('</div>')
    return ''.join(html)


def generate_timeseries_html(var_name, ts_data, max_rows):
    """Generate HTML for timeseries object"""
    
    time_arr = ts_data.get('Time', [])
    data_arr = ts_data.get('Data', [])
    
    if not isinstance(time_arr, np.ndarray):
        time_arr = np.array(time_arr)
    if not isinstance(data_arr, np.ndarray):
        data_arr = np.array(data_arr)
    
    # Flatten arrays if they are 2D (common for MATLAB data)
    time_arr = time_arr.flatten()
    data_arr = data_arr.flatten()
    
    total_rows = len(time_arr)
    show_all = total_rows <= max_rows
    display_rows = total_rows if show_all else max_rows
    
    # Create full data
    full_data = {'Time': time_arr.tolist(), 'Data': data_arr.tolist()}
    full_data_json = json.dumps(full_data)
    
    html = []
    html.append('<div class="stats">')
    html.append(f'<strong>Length:</strong> {total_rows} samples')
    html.append('</div>')
    
    html.append(f'<div class="array-container">')
    html.append(f'<table class="data-table" id="table_{var_name}">')
    html.append('<thead><tr><th>Index</th><th>Time</th><th>Data</th></tr></thead>')
    html.append('<tbody>')
    
    for i in range(display_rows):
        html.append(f'<tr><td>{i}</td><td>{time_arr[i]:.6g}</td><td>{data_arr[i]:.6g}</td></tr>')
    
    html.append('</tbody></table>')
    
    # Hidden data
    html.append(f'<script type="application/json" id="data_{var_name}">{full_data_json}</script>')
    
    if not show_all:
        hidden_count = total_rows - display_rows
        html.append(f'<p class="truncated-msg">... {hidden_count} more rows hidden ...</p>')
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="showAllRowsTS(\'{var_name}\')">Show All ({total_rows} samples)</button>')
        html.append(f'<button onclick="downloadCSV_TS(\'{var_name}\')">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', \'ts\')">Copy Data</button>')
        html.append(f'</div>')
    else:
        html.append(f'<div class="button-group">')
        html.append(f'<button onclick="downloadCSV_TS(\'{var_name}\')">Download CSV</button>')
        html.append(f'<button onclick="copyToClipboard(\'{var_name}\', \'ts\')">Copy Data</button>')
        html.append(f'</div>')
    
    html.append('</div>')
    return ''.join(html)


def generate_struct_html(var_name, struct_data, max_rows):
    """Generate HTML for struct"""
    
    html = ['<div class="struct-container"><ul class="struct-list">']
    
    for field_name, field_value in struct_data.items():
        html.append(f'<li><strong>{field_name}:</strong> ')
        
        if isinstance(field_value, np.ndarray):
            if field_value.size < 10:
                html.append(f'{field_value.tolist()}')
            else:
                html.append(f'Array{field_value.shape} ({field_value.dtype})')
        elif isinstance(field_value, (list, dict)):
            html.append(f'{str(field_value)[:200]}...')
        else:
            html.append(f'{field_value}')
        
        html.append('</li>')
    
    html.append('</ul></div>')
    return ''.join(html)


def generate_cell_array_html(var_name, cell_data, max_rows):
    """Generate HTML for cell array"""
    
    html = [f'<div class="cell-container">']
    html.append(f'<p><strong>Cell array with {len(cell_data)} elements</strong></p>')
    html.append('<ol class="cell-list">')
    
    display_count = min(len(cell_data), max_rows)
    
    for i in range(display_count):
        item = cell_data[i]
        html.append('<li>')
        
        if isinstance(item, np.ndarray):
            if item.size < 10:
                html.append(f'Array: {item.tolist()}')
            else:
                html.append(f'Array{item.shape} ({item.dtype})')
        elif isinstance(item, str):
            html.append(f'String: "{item}"')
        else:
            html.append(f'{type(item).__name__}: {str(item)[:100]}')
        
        html.append('</li>')
    
    if len(cell_data) > max_rows:
        html.append(f'<li><em>... {len(cell_data) - max_rows} more elements ...</em></li>')
    
    html.append('</ol></div>')
    return ''.join(html)


def get_html_header(mat_filename):
    """Return HTML header with CSS styling"""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MATLAB Variables: {mat_filename}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .var-type {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-weight: normal;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .summary-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .summary-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .summary-table a {{
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
        }}
        .summary-table a:hover {{
            text-decoration: underline;
        }}
        .ts-length {{
            color: #27ae60;
            font-weight: 500;
        }}
        .array-shape {{
            color: #8e44ad;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .variable-section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
            overflow-x: auto;
        }}
        .data-table th {{
            background-color: #34495e;
            color: white;
            padding: 10px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        .data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .data-table tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .data-table tbody tr:hover {{
            background-color: #e3f2fd;
        }}
        .truncated-msg {{
            color: #7f8c8d;
            font-style: italic;
            text-align: center;
            margin: 10px 0;
        }}
        .button-group {{
            margin: 15px 0;
            text-align: center;
        }}
        button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.3s;
        }}
        button:hover {{
            background-color: #2980b9;
        }}
        .string-value {{
            background-color: #fef9e7;
            padding: 15px;
            border-left: 4px solid #f39c12;
            font-family: 'Courier New', monospace;
        }}
        .struct-list, .cell-list {{
            background-color: #fafafa;
            padding: 15px 30px;
            border-radius: 4px;
        }}
        .struct-list li, .cell-list li {{
            margin: 8px 0;
        }}
        .error {{
            color: #e74c3c;
            background-color: #fadbd8;
            padding: 10px;
            border-radius: 4px;
        }}
        .scalar-value {{
            font-size: 1.2em;
            padding: 10px;
            background-color: #e8f6f3;
            border-radius: 4px;
        }}
        .empty-array {{
            color: #7f8c8d;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }}
        .search-container {{
            margin: 20px 0;
        }}
        #searchBox {{
            width: 100%;
            padding: 12px 20px;
            font-size: 1em;
            border: 2px solid #3498db;
            border-radius: 4px;
        }}
        .array-container {{
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>MATLAB Variables: {mat_filename}</h1>
'''


def get_javascript():
    """Return JavaScript for interactive features"""
    return '''
<script>
function filterVariables() {
    const input = document.getElementById('searchBox');
    const filter = input.value.toLowerCase();
    const sections = document.getElementsByClassName('variable-section');
    
    for (let section of sections) {
        const varName = section.getAttribute('data-varname').toLowerCase();
        if (varName.includes(filter)) {
            section.style.display = '';
        } else {
            section.style.display = 'none';
        }
    }
}

function showAllRows(varName, dims) {
    const dataElement = document.getElementById('data_' + varName);
    const tableElement = document.getElementById('table_' + varName);
    
    if (!dataElement || !tableElement) return;
    
    const data = JSON.parse(dataElement.textContent);
    const tbody = tableElement.getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    if (dims === 1) {
        // 1D array
        for (let i = 0; i < data.length; i++) {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = i;
            row.insertCell(1).textContent = data[i];
        }
    } else if (dims === 2) {
        // 2D array
        const numCols = data[0].length;
        for (let i = 0; i < data.length; i++) {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = i;
            for (let j = 0; j < numCols; j++) {
                row.insertCell(j + 1).textContent = data[i][j].toPrecision(6);
            }
        }
    }
    
    // Remove truncation message and show all button
    const container = tableElement.parentElement;
    const truncMsg = container.getElementsByClassName('truncated-msg')[0];
    if (truncMsg) truncMsg.remove();
    
    const buttons = container.getElementsByTagName('button');
    for (let btn of buttons) {
        if (btn.textContent.startsWith('Show All')) {
            btn.remove();
            break;
        }
    }
}

function showAllRowsTS(varName) {
    const dataElement = document.getElementById('data_' + varName);
    const tableElement = document.getElementById('table_' + varName);
    
    if (!dataElement || !tableElement) return;
    
    const data = JSON.parse(dataElement.textContent);
    const tbody = tableElement.getElementsByTagName('tbody')[0];
    tbody.innerHTML = '';
    
    for (let i = 0; i < data.Time.length; i++) {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = i;
        row.insertCell(1).textContent = data.Time[i].toPrecision(6);
        row.insertCell(2).textContent = data.Data[i].toPrecision(6);
    }
    
    // Remove truncation message
    const container = tableElement.parentElement;
    const truncMsg = container.getElementsByClassName('truncated-msg')[0];
    if (truncMsg) truncMsg.remove();
    
    const buttons = container.getElementsByTagName('button');
    for (let btn of buttons) {
        if (btn.textContent.startsWith('Show All')) {
            btn.remove();
            break;
        }
    }
}

function downloadCSV(varName, dims) {
    const dataElement = document.getElementById('data_' + varName);
    if (!dataElement) return;
    
    const data = JSON.parse(dataElement.textContent);
    let csv = '';
    
    if (dims === 1) {
        csv = 'Index,Value\\n';
        for (let i = 0; i < data.length; i++) {
            csv += i + ',' + data[i] + '\\n';
        }
    } else if (dims === 2) {
        // Header
        csv = 'Row,' + Array.from({length: data[0].length}, (_, i) => i).join(',') + '\\n';
        // Data
        for (let i = 0; i < data.length; i++) {
            csv += i + ',' + data[i].join(',') + '\\n';
        }
    }
    
    downloadFile(csv, varName + '.csv', 'text/csv');
}

function downloadCSV_TS(varName) {
    const dataElement = document.getElementById('data_' + varName);
    if (!dataElement) return;
    
    const data = JSON.parse(dataElement.textContent);
    let csv = 'Index,Time,Data\\n';
    
    for (let i = 0; i < data.Time.length; i++) {
        csv += i + ',' + data.Time[i] + ',' + data.Data[i] + '\\n';
    }
    
    downloadFile(csv, varName + '_timeseries.csv', 'text/csv');
}

function copyToClipboard(varName, dims) {
    const dataElement = document.getElementById('data_' + varName);
    if (!dataElement) return;
    
    const data = JSON.parse(dataElement.textContent);
    let text = '';
    
    if (dims === 1) {
        text = data.join('\\n');
    } else if (dims === 2) {
        text = data.map(row => row.join('\\t')).join('\\n');
    } else if (dims === 'ts') {
        text = 'Time\\tData\\n';
        for (let i = 0; i < data.Time.length; i++) {
            text += data.Time[i] + '\\t' + data.Data[i] + '\\n';
        }
    }
    
    navigator.clipboard.writeText(text).then(() => {
        alert('Data copied to clipboard!');
    });
}

function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}
</script>
'''


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
        html_file = sys.argv[2] if len(sys.argv) > 2 else None
        export_matlab_to_html(mat_file, html_file)
    else:
        # Test with example file
        export_matlab_to_html("mixed_signals_1.mat")
