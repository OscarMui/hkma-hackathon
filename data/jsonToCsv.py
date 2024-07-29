import json
import csv

# Load the JSON data
with open('ai.json', 'r') as f:
    data = json.load(f)["result"]["records"]

# Define the fieldnames for the CSV file
fieldnames = list(data[0].keys())

# Open the CSV file and write the data
with open('ai.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)