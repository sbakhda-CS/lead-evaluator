import json
import csv


def get_cols(file_name):

    f = open(file_name, 'r')
    text = f.read()
    text = text[text.index('\n')+1::]
    f.close()

    heads = []
    with open(file_name, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            heads = row
            break
    f = open('data.csv', 'w')
    f.write(text)
    return heads


if __name__ == "__main__":
    cols = get_cols("data.csv")

    init_json = {
        "name": "cortex/mymodel-data-type",
        "camel": "1.0.0",
        "title": "mymodel Data Type",
        "parameters": []
    }

    for col in cols:
        init_json['parameters'].append({"name": col, "type": "number"})

    f = open('training-data-type.json', 'w')
    f.write(json.dumps(init_json))
    f.close()