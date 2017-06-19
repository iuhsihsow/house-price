import csv

file_name = "../data/beijing_house_price.csv"
record_count = 0

max_price = 0
min_price = 10000000000
with open(file_name, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        record_count += 1
        print(row['id'], row['X'], row['Y'], row['price'])
        max_price = max((int)(row['price']), max_price)
        min_price = min((int)(row['price']), min_price)

    print(record_count, max_price, min_price)