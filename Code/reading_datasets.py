import pickle
import json
import csv

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_task(location, split = 'train'):
    filename = location + split + '.tsv'
    
    # Creating empty array for combined data
    data = []

    # Creating 
    fm_data = []
    saho_data = []
    sc_data = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i > 0:
                tweet_id = row[0]
                sentence = row[1].strip()
                label = row[3]
                topic = row[2]
                premise = row[4]
                

                if topic == 'face masks':
                    fm_data.append((sentence, label, topic, tweet_id, premise))
                elif topic == 'stay at home orders':
                    saho_data.append((sentence, label, topic, tweet_id, premise))
                elif topic == 'school closures':
                    sc_data.append((sentence, label, topic, tweet_id, premise))


    data = [fm_data, saho_data, sc_data]

    return data


if __name__ == '__main__':
    location = '../Datasets/'
    split = 'train'
    
    data = read_task(location, split)
    print(len(data))

    data = read_task(location, 'dev')
    print(len(data))

    




