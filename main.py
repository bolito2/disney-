import csv

with open('cannabis.csv', newline='') as csvfile:
    cannabis_data = csv.reader(csvfile)
    cannabis_names = []

    # The first column of the data contains the strain name
    for row in cannabis_data:
        cannabis_names.append(row[0])

    # First row is metadata so delete it
    cannabis_names = cannabis_names[1:]

