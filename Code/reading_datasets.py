import pickle
import json
import csv

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_task7(location, split = 'train'):
    filename = location + split + '.csv'

    data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i > 0:
                tweet_id = row[0]
                sentence = row[1].strip()
                label = row[2]
                data.append((sentence, label))

    return data


if __name__ == '__main__':
    location = '../Datasets/TASK7/'
    split = 'train'
    
    data = read_task7(location, split)
    print(len(data))

    data = read_task7(location, 'dev')
    print(len(data))

    




