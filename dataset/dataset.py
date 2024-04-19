import pandas as pd

with open('raw_data.tsv', 'r', encoding='utf-8') as file:
    error_count = 0
    correct_count = 0
    error_list = []
    correct_list = []
    for line in file:
        fields = line.strip().split('\t')
        if float(fields[4]) <= 50 and error_count < 300:
            error_count += 1
            error_list.append([fields[1], fields[2], 0])
        elif float(fields[4]) >= 80 and correct_count < 300:
            correct_count += 1
            correct_list.append([fields[1], fields[2], 1])
        if error_count == 300 and correct_count == 300:
            break

combined_df = pd.DataFrame(error_list + correct_list)
combined_df.to_excel('data.xlsx', index=False, header=False)
