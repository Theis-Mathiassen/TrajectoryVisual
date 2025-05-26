import re
import csv
import argparse
from collections import defaultdict

def extract_config_from_filename(filename):
    """
    Extracts (method, setting) from a filename like 'scores_knnX_rangeY_simZ.pkl'.
    Example: 'scores_knn1_range4_simc+f.pkl' -> ('Grad', 'c+f')
    """
    match = re.search(r"range(\d+)_sim([a-zA-Z0-9+]+)\.pkl", filename)
    if not match:
        return None

    range_num = match.group(1)
    sim_type = match.group(2)

    method_map = {
        "4": "Grad",
        "3": "SA",
        "2": "SO",
        "1": "WTA",
    }
    setting_map = {
        "c+f": "c+f",
        "a": "a",
        "c": "c",
        "m": "m",
    }

    method = method_map.get(range_num)
    setting = setting_map.get(sim_type)

    if method and setting:
        return (method, setting)
    return None

def generate_latex_table(data_rows, csv_filepath):
    """
    Generates a LaTeX table from the provided data rows (list of dictionaries).
    Each dictionary represents a row from the CSV with column headers as keys.
    """
    grouped_data = defaultdict(list)
    
    # Define expected column names for clarity and checking
    expected_columns = ['folder', 'file', 'cr', 'f1_1', 'f1_2', 'f1_3', 'f1_4']
    if not data_rows:
        print("Warning: No data rows provided to generate_latex_table.")
        # Fallback to create an empty table structure or return an error message
        # For now, we'll let it proceed and it will likely produce an empty data section
    elif not all(col in data_rows[0] for col in expected_columns):
        print(f"Error: CSV data is missing one or more expected columns: {expected_columns}")
        print(f"Available columns: {list(data_rows[0].keys())}")
        print("Please ensure your CSV file contains these columns in its header.")
        return "% Error: Missing required columns in CSV input. See console output."


    for row_num, row in enumerate(data_rows):
        try:
            folder_name = row['folder']
            file_name = row['file']
            cr = float(row['cr'])
            f1_avg = float(row['f1_1'])    # Maps to Avg
            f1_range = float(row['f1_2'])  # Maps to Range
            f1_sim = float(row['f1_3'])    # Maps to Sim
            f1_knn = float(row['f1_4'])    # Maps to Knn
            
            config = extract_config_from_filename(file_name)
            if config:
                grouped_data[config].append([cr, f1_avg, f1_range, f1_sim, f1_knn])
        except ValueError as e:
            print(f"Warning: Skipping row {row_num + 1} due to data conversion error (ValueError: {e}): {row}")
            continue
        except KeyError as e:
            print(f"Warning: Skipping row {row_num + 1} due to missing column (KeyError: {e}). Ensure columns {expected_columns} exist.")
            continue


    for config_key in grouped_data:
        grouped_data[config_key].sort(key=lambda x: x[0])

    publication_order = [("Grad", "a"), ("Grad", "c"), ("Grad", "c+f"), ("Grad", "m"),
        ("SA", "a"), ("SA", "c"), ("SA", "c+f"), ("SA", "m"),
        ("SO", "a"), ("SO", "c"), ("SO", "c+f"), ("SO", "m"),
        ("WTA", "a"), ("WTA", "c"),("WTA", "c+f"), ("WTA", "m")
    ]

    latex_output = []
    latex_output.append(r"\begin{longtable}{lS[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]S[table-format=1.3]}")
    latex_output.append(r"  \caption{Experimental Results: F1 Scores for Different Configurations and Compression Rates (CR)}")
    latex_output.append(r"  \label{tab:experimental_results_eval_geolife_gauss.csv} \\ ")
    latex_output.append(r"  \toprule")
    latex_output.append(r"  {Configuration} & {\text{CR}} & \multicolumn{4}{c}{F1 Scores} \\")
    latex_output.append(r"  \cmidrule(lr){3-6}")
    latex_output.append(r"  & & {\text{Avg}} & {\text{Range}} & {\text{Sim}} & {\text{Knn}} \\")
    latex_output.append(r"  \midrule")

    first_group_printed = False
    for config_tuple in publication_order:
        if config_tuple in grouped_data:
            config_data_rows = grouped_data[config_tuple]
            if len(config_data_rows) != 5:
                print(f"Warning: Configuration {config_tuple} has {len(config_data_rows)} entries, expected 5. Skipping or partially printing.")
                # Decide how to handle: skip, or print if at least one row exists? For now, requires 5.
                if not config_data_rows: continue # Skip if empty

            if first_group_printed: # Add midrule before new group, except for the very first one
                 latex_output.append(r"    \midrule")
            
            method_name, setting_name = config_tuple
            multirow_label = f"{method_name}, {setting_name}"
            
            # For the first row of a multirow group
            if config_data_rows: # Check if there's any data to print
                first_row_data = config_data_rows[0]
                cr, f1_avg, f1_range, f1_sim, f1_knn = first_row_data
                latex_output.append(f"    \\multirow{{5}}{{*}}{{\\makecell[l]{{{multirow_label}}}}} & {cr:.3f} & {f1_avg:.15f} & {f1_range:.15f} & {f1_sim:.15f} & {f1_knn:.15f} \\\\")
                
                # For subsequent rows in the same group
                for i in range(1, len(config_data_rows)):
                    row_data = config_data_rows[i]
                    cr, f1_avg, f1_range, f1_sim, f1_knn = row_data
                    latex_output.append(f"    & {cr:.3f} & {f1_avg:.15f} & {f1_range:.15f} & {f1_sim:.15f} & {f1_knn:.15f} \\\\")
                
                # If there are fewer than 5 rows, add empty rows to complete the multirow span if desired
                # For simplicity, this version assumes 5 rows or prints what it has.
                # The \multirow{5} might lead to visual issues if not exactly 5 rows are printed.
                # A more robust solution would adjust \multirow number or fill with empty content.
                # Current code relies on the warning and `if len(config_data_rows) != 5:` check.
                # If strictly 5 rows are needed:
                if len(config_data_rows) != 5:
                     print(f"Strict check: Configuration {config_tuple} did not yield 5 rows of data after processing. LaTeX output might be misaligned for this group.")

            first_group_printed = True
        else:
            print(f"Notice: Configuration {config_tuple} not found in the provided CSV data.")


    latex_output.append(r"    \bottomrule")
    latex_output.append(r"\end{longtable}")

    return "\n".join(latex_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a LaTeX table from CSV data.")
    parser.add_argument("csv_filepath", help="Path to the input CSV file.")
    
    args = parser.parse_args()

    data_rows = []
    try:
        with open(args.csv_filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                print(f"Error: CSV file '{args.csv_filepath}' appears to be empty or has no header row.")
                exit(1)
            
            # Basic check for expected column names based on the problem description
            # The image shows columns: folder, cr, f1_1, f1_2, f1_3, f1_4 ...
            expected_csv_cols = ['folder', 'cr', 'f1_1', 'f1_2', 'f1_3', 'f1_4']
            missing_cols = [col for col in expected_csv_cols if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: The CSV file '{args.csv_filepath}' is missing the following required columns: {', '.join(missing_cols)}.")
                print(f"Available columns in CSV: {', '.join(reader.fieldnames)}")
                exit(1)

            for row in reader:
                data_rows.append(row)
                
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{args.csv_filepath}'")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        exit(1)

    if not data_rows:
        print(f"No data successfully read from '{args.csv_filepath}'. Cannot generate table.")
        exit(1)
        
    latex_code = generate_latex_table(data_rows, args.csv_filepath)
    
    print("\n--- Generated LaTeX Code ---")
    print(latex_code)

    print(r"""
--- LaTeX Preamble Requirements ---
For the generated LaTeX code to compile correctly, you might need to include the following
packages in your LaTeX document's preamble:
\usepackage{multirow}     % For \multirow
\usepackage{makecell}     % For \makecell
\usepackage{booktabs}     % For \toprule, \midrule, \bottomrule
\usepackage{siunitx}      % For S column type and number formatting, and \sisetup
""")