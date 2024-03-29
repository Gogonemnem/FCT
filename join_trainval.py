import json

with open('/home/bibahaduri/dota_dataset/coco/annotations/instances_train2017.json', 'r') as f:
    train_data = json.load(f)

with open('/home/bibahaduri/dota_dataset/coco/annotations/instances_val2017.json', 'r') as f:
    val_data = json.load(f)

trainval_data = {
    "images": train_data["images"] + val_data["images"],
    "annotations": train_data["annotations"] + val_data["annotations"],
    # You may need to update other fields like licenses, categories, etc.
    "licenses": train_data["licenses"],
    "info": train_data["info"],
    "categories": train_data["categories"]
}

# Example function to update IDs
def update_ids(data, id_offset):
    for item in data["images"]:
        item["id"] += id_offset
    for item in data["annotations"]:
        item["id"] += id_offset
        item["image_id"] += id_offset

# Calculate the offset based on the maximum existing IDs
max_train_id = max(item["id"] for item in train_data["images"] + train_data["annotations"])
id_offset = max_train_id + 1

# Update IDs in val_data
update_ids(val_data, id_offset)

trainval_data = {
    "images": train_data["images"] + val_data["images"],
    "annotations": train_data["annotations"] + val_data["annotations"],
    "licenses": train_data["licenses"],
    "info": train_data["info"],
    "categories": train_data["categories"]
}

with open('instances_trainval2017.json', 'w') as f:
    json.dump(trainval_data, f)