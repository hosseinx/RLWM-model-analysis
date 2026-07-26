import pandas as pd
import os
from collections import defaultdict

# --- Initial Settings ---
# The path to the folder containing the data files
DATA_PATH = 'data/my_experiments/'

def calculate_features(df):
    """
    This function calculates the computational columns for a given DataFrame.
    """
    # Ensure the data is sorted by its original order
    df = df.sort_index()

    # 1. Calculate trial_iter (trial counter starting from 1)
    df['trial_iter'] = range(1, len(df) + 1)

    # 2. Calculate st_iter (stimulus iteration counter starting from 1)
    df['st_iter'] = df.groupby('Stimulus_Pair').cumcount() + 1

    # Prepare lists for the more complex columns
    rpreds = []
    delays = []
    pers_errors = []
    
    # Dictionaries to keep track of the state for each stimulus
    last_correct_response_index = {}
    correct_response_counts = defaultdict(int)
    past_errors = defaultdict(set)
    
    # 3. Loop through rows to calculate rpred, delay, and pers
    for index, row in df.iterrows():
        stimulus = row['Stimulus_Pair']
        action = row['response']
        is_correct = row['correct'] == 1

        # Calculate rpred (count of *previous* correct responses)
        rpreds.append(correct_response_counts[stimulus])
        
        # Calculate delay (distance from the last correct response)
        if stimulus in last_correct_response_index:
            delays.append(index - last_correct_response_index[stimulus])
        else:
            delays.append(0) # You can use np.nan here if you prefer
            
        # Calculate pers (perseveration error)
        is_perseveration = 0
        if not is_correct:
            if action in past_errors[stimulus]:
                is_perseveration = 1
            past_errors[stimulus].add(action)
        pers_errors.append(is_perseveration)

        # Update the tracking dictionaries *after* calculating the current row's values
        if is_correct:
            correct_response_counts[stimulus] += 1
            last_correct_response_index[stimulus] = index
            
    df['rpred'] = rpreds
    df['delay'] = delays
    df['pers'] = pers_errors
    
    return df

def process_all_parte1_files():
    """
    Finds all 'Parte 1' files in the data folder and adds the computational columns to them.
    """
    files_processed = 0
    # Search the data folder for relevant files
    for filename in os.listdir(DATA_PATH):
        if 'RLWM-Parte 1_' in filename and filename.endswith('.csv'):
            file_path = os.path.join(DATA_PATH, filename)
            
            try:
                print(f"Processing file: {filename} ...")
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Calculate the new columns
                df_updated = calculate_features(df)
                
                # Overwrite the original file with the new data
                df_updated.to_csv(file_path, index=False)
                
                print(f"Successfully updated file: {filename}")
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if files_processed == 0:
        print("No files found to process. Please check the file path and names.")
    else:
        print(f"\nOperation completed successfully. {files_processed} files were updated.")

if __name__ == '__main__':
    process_all_parte1_files()