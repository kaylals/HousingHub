import re
import os
import pdfplumber
import csv
import pandas as pd
import numpy as np



# Define input and output directories
input_dir = 'data/individual_level/raw_pdf_700'
output_dir = 'data/individual_level/raw_csv_700'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to clean and parse the text into rows and columns
def clean_and_parse_text(text):
    cleaned_data = []
    lines = text.split('\n')
    
    for line in lines:
        # Remove unwanted footer information
        if 'Matrix' in line or 'https://' in line:
            continue
        
        # Extract relevant data, assuming each relevant line has a date at the beginning
        if line and line[0].isdigit():
            # Split the line into columns
            columns = line.split()
            
            # Initialize an empty list to store the processed columns
            processed_columns = []
            
            # Process the columns
            i = 0
            while i < len(columns):
                if i == 4:  # Address starts at index 4
                    # Combine address parts until we reach AreaCity
                    address_parts = []
                    while i < len(columns) and not columns[i].endswith('700Seattle'):
                        address_parts.append(columns[i])
                        i += 1
                    processed_columns.append(' '.join(address_parts))
                    if i < len(columns):
                        processed_columns.append(columns[i])  # AreaCity
                        i += 1
                    else:
                        processed_columns.append('700Seattle')  # Default AreaCity if not found
                else:
                    processed_columns.append(columns[i])
                    i += 1
            
            cleaned_data.append(processed_columns)
    
    return cleaned_data

# Function to process a single PDF file
def process_pdf(pdf_path, csv_path):
    # Extract text from the PDF using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        pdf_text = []
        for page in pdf.pages:
            page_text = page.extract_text()
            pdf_text.append(page_text)

    # Clean and parse the text
    parsed_data = [clean_and_parse_text(page) for page in pdf_text]
    parsed_data = [item for sublist in parsed_data for item in sublist]  # Flatten the list

    # Define column names
    column_names = [
        "#", "MLS", "Stat", "Type", "Address", "AreaCity", "State", 
        "List/Sell $", "CDOMBds", "Bths", "SF", "Stat Date"
    ]

    # Save the cleaned data to a CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(column_names)  # Write column names as header
        for row in parsed_data:
            if len(row) == len(column_names):  # Ensure row has correct number of columns
                csvwriter.writerow(row)
            else:
                print(f"Skipping row with incorrect number of columns: {row}")

    # Post-process the CSV file
    df = pd.read_csv(csv_path)

    def safe_int(x):
        try:
            return int(x)
        except ValueError:
            return np.nan

    # Separate AreaCity into Area and City
    df['Area'] = df['AreaCity'].astype(str).str.extract(r'(\d+)', expand=False).apply(safe_int)
    df['City'] = df['AreaCity'].astype(str).str.extract(r'(\D+)', expand=False)

    # Separate CDOMBds into CDOM and Bds
    df['CDOMBds'] = df['CDOMBds'].astype(str)
    df['CDOM'] = df['CDOMBds'].str[:-1].apply(safe_int)
    df['Bds'] = df['CDOMBds'].str[-1].apply(safe_int)

    # Drop the original columns
    df = df.drop(['AreaCity', 'CDOMBds'], axis=1)

    # Save the updated dataframe back to the CSV file
    df.to_csv(csv_path, index=False)

    print(f'Processed and saved: {csv_path}')

# Process all PDF files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_dir, filename)
        csv_name = filename.replace('.pdf', '.csv')
        csv_path = os.path.join(output_dir, csv_name)
        process_pdf(pdf_path, csv_path)

print("All PDF files have been processed and converted to CSV.")