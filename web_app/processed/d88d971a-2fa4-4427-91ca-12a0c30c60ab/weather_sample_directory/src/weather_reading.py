import csv

# Reads a CSV file and returns its content as a list of rows
def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
    return data

# No error handling for missing values
data_2018 = read_csv('data/weather_2018.csv')
data_2019 = read_csv('data/weather_2019.csv')
data_2020 = read_csv('data/weather_2020.csv')

# Printing data to debug but no actual logging or proper debugging methods
print("Data 2018:", data_2018)
print("Data 2019:", data_2019)