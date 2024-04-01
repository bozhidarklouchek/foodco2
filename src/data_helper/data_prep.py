import csv, random, json

def split_data(input_data):
    total_items = len(input_data)
    
    # Calculate sizes for each data
    train_size = int(total_items * 0.72)
    val_size = int(total_items * 0.18)
    
    # Generate empty collections
    train = []
    val = []
    test = []
    
    # Distribute items into collections
    for i, item in enumerate(input_data):
        if i < train_size:
            train.append(item)
        elif i < train_size + val_size:
            val.append(item)
        else:
            test.append(item)
    
    return train, val, test

def csv_to_3tsv(seed, csv_file):
    triplets = []

    # Read triplets from csv
    with open(csv_file, 'r', newline='', encoding='utf-8') as csv_input:
        csv_reader = csv.DictReader(csv_input)
        for row in csv_reader:
            triplets.append(row)
        
    random.seed(seed)
    random.shuffle(triplets)

    tr, val, test = split_data(triplets)

    print(len(tr))
    print(len(val))
    print(len(test))

    with open('data/train.tsv', 'w', newline='', encoding='utf-8') as tsv_output:
        tsv_writer = csv.writer(tsv_output, delimiter='\t')
        for row in tr:
            data_to_write = [row['anchor'], row['pos'], row['neg']]
            data_to_write = [data_point.replace('\n', ' ').replace(',', '') for data_point in data_to_write if len(data_point) != 0]
            tsv_writer.writerow(data_to_write)

    with open('data/dev.tsv', 'w', newline='', encoding='utf-8') as tsv_output:
        tsv_writer = csv.writer(tsv_output, delimiter='\t')
        for row in val:
            data_to_write = [row['anchor'], row['pos'], row['neg']]
            data_to_write = [data_point.replace('\n', ' ').replace(',', '') for data_point in data_to_write if len(data_point) != 0]
            tsv_writer.writerow(data_to_write)

    with open('data/test.csv', 'w', newline='', encoding='utf-8') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(['anchor', 'anchor_full_ingred_name', 'anchor_matched_ingred_name',
                         'pos', 'pos_full_ingred_name', 'pos_matched_ingred_name',
                         'neg', 'neg_raw_ingred_name'])
        csv_writer.writerows([sample.values() for sample in test])

def place_all_ingreds_in_instruction(instruction):
    # Read all ingredients
    ingreds = set()
    with open('C:/Users/klouc/Desktop/FOODCO2/data/KB.json', 'r', encoding='utf-8') as file:
    # Load JSON data from the file
        data = json.load(file)
        for ingred in data:
            ingreds.add(ingred['ingredient'])
    
    # Save all instruction versions to .tsv
    with open('C:/Users/klouc/Desktop/FOODCO2/data/instructions.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        for ingred in ingreds:
            writer.writerow([instruction.replace('{0}', ingred.lower())])


# TO MAKE TRIPLETS
seed = 42
csv_file = 'data/triplets.csv'
csv_to_3tsv(seed, csv_file)
print(f"Conversion from '{csv_file}' to .tsv completed successfully.")

# TO MAKE EMBED TEST
# place_all_ingreds_in_instruction("Chicken fajitas with guacamole: Add the {0}.")        
