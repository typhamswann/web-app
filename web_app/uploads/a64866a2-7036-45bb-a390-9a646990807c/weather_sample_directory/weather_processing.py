import csv

# Processes weather data for averages but doesn't handle missing data
def process_weather(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skipping the header
        total_temp = 0
        total_humidity = 0
        count = 0
        for row in reader:
            total_temp += int(row[1])
            total_humidity += int(row[2])
            count += 1
        
        avg_temp = total_temp / count
        avg_humidity = total_humidity / count

    # No return value, just prints
    print(f"Average Temperature: {avg_temp}")
    print(f"Average Humidity: {avg_humidity}")

process_weather('data/weather_2018.csv')
process_weather('data/weather_2019.csv')
