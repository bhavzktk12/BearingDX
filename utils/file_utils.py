import os
import csv
import json
import pandas as pd
import xml.etree.ElementTree as ET

def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return 'csv'
    elif ext == '.json':
        return 'json'
    elif ext == '.xml':
        return 'xml'
    elif ext in ['.xls', '.xlsx']:
        return 'excel'
    elif ext == '.txt':
        return 'txt'
    elif ext == '.bin':
        return 'binary'
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def read_csv(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def read_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
        return data if isinstance(data, list) else [data]

def read_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_dict(orient='records')

def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return [
        {child.tag: child.text for child in entry}
        for entry in root
    ]

def read_txt(file_path):
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        return [{'line': line.strip()} for line in lines]

def read_binary(file_path):
    with open(file_path, 'rb') as f:
        binary_data = f.read()
        return [{'raw_bytes': binary_data.hex()}]
