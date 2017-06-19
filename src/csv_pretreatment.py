import csv

x_field = 'X'
y_field = 'Y'
id_field = 'id'
price_field = 'price'

input_file = "../data/beijing_house_price.csv"
output_file = "../data/beijing_x_y_price.csv"

record_count = 0

x_col = []
y_col = []
id_col = []
price_col = []

with open(input_file, 'rt', encoding='utf-8') as ifile:
    reader = csv.DictReader(ifile)
    rows = [row for row in reader]
    for row in reader:
        record_count += 1
        id_col.append(row['id'])
        x_col.append(row[x_field])
        y_col.append(row[y_field])
        id_col.append(row[id_field])
        price_col.append(row[price_field])

    with open(output_file, 'wt') as ofile:
        field_names = [id_field, x_field, y_field, price_field]
        writer = csv.DictWriter(ofile, fieldnames=field_names, delimiter=',', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({id_field : row[id_field],
                             x_field : row[x_field],
                             y_field : row[y_field],
                             price_field : row[price_field],
                             })





