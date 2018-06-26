import json
import csv


def get_cols(file_name):
    with open(file_name, newline='\n') as f:
        reader = csv.reader(f)
        for row in reader:
            return row


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