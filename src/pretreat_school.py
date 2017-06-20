import csv

school_file = "../data/vipschool.csv"

school_names_array = []

record_count = 0
with open(school_file, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        school_names_array.append(row['name'])

school_names_array.append("双榆小学")
school_names_array.append("芳草地")

print(len(school_names_array))

def has_vip_school(schools):
    for school in school_names_array:
        if school in schools:
            return True
    return False


result_array = []

vip_count = 0
nvip_count = 0
file_name = "../data/in6ring.csv"
with open(file_name, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        schools = row['school']
        contain_vip_school = has_vip_school(schools)
        if contain_vip_school:
            result_array.append('1')
            vip_count += 1
        else:
            #result_array.append(schools)
            result_array.append('0')
            nvip_count += 1

with open("../data/school_result.csv", 'wt') as file:
    for res in result_array:
        file.write(res + "\n")
print("VIP:" + str(vip_count) + "\nNot:" + str(nvip_count))




