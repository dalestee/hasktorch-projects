import csv
import os
import sys

def copy_non_missing_rows(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-cp.csv")

    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        writer.writerow(next(reader))

        for row in reader:
            if '' not in row:
                writer.writerow(row)

def removeCol(input_path, col):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-rmv.csv")

    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        header = next(reader)
        header.pop(col)
        writer.writerow(header)

        for row in reader:
            row.pop(col)
            writer.writerow(row)

def fill_missing_values(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-filled.csv")

    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        writer.writerow(next(reader))

        for row in reader:
            for i, value in enumerate(row):
                if value == '' and i != 2 and i != 9 and i != 10 and i != 3 and i != 7:
                    row[i] = median_of_column(input_path, i)
            writer.writerow(row)

def median_of_column(input_path, column_nb):
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        values = []
        next(reader)

        for row in reader:
            if row[column_nb] != '':
                values.append(float(row[column_nb]))
        values.sort()
        n = len(values)
        if n % 2 == 0:
            return (values[n // 2 - 1] + values[n // 2]) / 2
        else:
            return values[n // 2]

def changeSexIntoBinary(input_path):
    compte = 0
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-sex.csv")

    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        writer.writerow(next(reader))

        for row in reader:
            if row[2] == '' and compte == 0:
                row[2] = compte
                compte = 1
            elif row[2] == '' and compte == 1:
                row[2] = compte
                compte = 0
            elif row[2] == 'male':
                row[2] = 0
            else:
                row[2] = 1
            writer.writerow(row)

def changeEmbarkedIntoNumeric(input_path):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-embarked.csv")

    with open(input_path, 'r') as input_file, open(output_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        writer.writerow(next(reader))

        for row in reader:
            if row[8] == 'C':
                row[8] = 1
            elif row[8] == 'Q':
                row[8] = 2
            else:
                row[8] = 3
            writer.writerow(row)

def normalize_csv(input_path):

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(os.path.dirname(input_path), f"{base_name}-normalized.csv")

    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        data = list(reader)

    # Transpose the data to work with columns
    data = list(map(list, zip(*data)))

    # Normalize all columns
    normalized_data = []
    for column in data:
        min_value = float(min(column[1:]))  # Exclude header
        max_value = float(max(column[1:]))  # Exclude header
        range_value = max_value - min_value
        if range_value == 0:
            normalized_column = [column[0]] + [0 for _ in column[1:]]  # Set all normalized values to 0
        else:
            normalized_column = [column[0]] + [(float(value) - min_value) / range_value for value in column[1:]]
        normalized_data.append(normalized_column)

    # Transpose the data back to rows
    normalized_data = list(map(list, zip(*normalized_data)))

    # Write the normalized data back to a CSV file
    with open(output_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(normalized_data)

if __name__ == "__main__":
    input_path = sys.argv[1]
    removeCol(input_path, 0)

