import pandas as pd
import json
import random

def train_val_test_split(total_size, train_split=0.8, val_split=0.1, test_split=0.1, sample_size=1000):
    random.seed(9)
    indices = random.sample(range(0, total_size), sample_size)
    train_size = int(sample_size * train_split)
    val_size =  int(sample_size * val_split)
    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices

def csv_to_json(csv_file_path1, csv_file_path2):
    # Read and process the first CSV file
    df1 = pd.read_csv(csv_file_path1, delimiter='|')
    df1.columns = df1.columns.str.strip()
    df1 = df1.rename(columns={'image_name': 'image', 'comment': 'caption'})

    image_names = list(set(df1['image']))
    train_indices, val_indices, test_indices = train_val_test_split(len(image_names))

    train_image_names = [image_names[i] for i in train_indices]
    val_image_names = [image_names[i] for i in val_indices]
    test_image_names = [image_names[i] for i in test_indices]

    train_df1 = df1[df1['image'].isin(train_image_names)]
    train_json1 = train_df1[['image', 'caption']].to_dict(orient='records')

    val_df1 = df1[df1['image'].isin(val_image_names)]
    val_grouped1 = val_df1.groupby('image')['caption'].apply(list).reset_index()
    val_json1 = val_grouped1[['image', 'caption']].to_dict(orient='records')

    test_df1 = df1[df1['image'].isin(test_image_names)]
    test_grouped1 = test_df1.groupby('image')['caption'].apply(list).reset_index()
    test_json1 = test_grouped1[['image', 'caption']].to_dict(orient='records')

    # Read and process the second CSV file
    df2 = pd.read_csv(csv_file_path2, delimiter='|')
    df2.columns = df2.columns.str.strip()
    df2 = df2.rename(columns={'image_name': 'image', 'comment': 'caption'})

    train_df2 = df2[df2['image'].isin(train_image_names)]
    train_json2 = train_df2[['image', 'caption']].to_dict(orient='records')

    val_df2 = df2[df2['image'].isin(val_image_names)]
    val_grouped2 = val_df2.groupby('image')['caption'].apply(list).reset_index()
    val_json2 = val_grouped2[['image', 'caption']].to_dict(orient='records')

    test_df2 = df2[df2['image'].isin(test_image_names)]
    test_grouped2 = test_df2.groupby('image')['caption'].apply(list).reset_index()
    test_json2 = test_grouped2[['image', 'caption']].to_dict(orient='records')

    return train_json1, val_json1, test_json1, train_json2, val_json2, test_json2

def save_to_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


csv_file_path1 = '../output_captions.csv'
train_file_path1 = './f30k_butd_rand800_train.json'
val_file_path1 = './f30k_butd_rand100_val.json'
test_file_path1 = './f30k_butd_rand100_test.json'

csv_file_path2 = './results.csv'
train_file_path2 = './f30k_human_rand800_train.json'
val_file_path2 = './f30k_human_rand100_val.json'
test_file_path2 = './f30k_human_rand100_test.json'

train_json1, val_json1, test_json1, train_json2, val_json2, test_json2 = csv_to_json(csv_file_path1, csv_file_path2)
save_to_json(train_json1, train_file_path1)
save_to_json(val_json1, val_file_path1)
save_to_json(test_json1, test_file_path1)
save_to_json(train_json2, train_file_path2)
save_to_json(val_json2, val_file_path2)
save_to_json(test_json2, test_file_path2)
