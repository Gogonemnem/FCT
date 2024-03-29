import json

# Load the existing annotations JSON file
with open('/home/bibahaduri/pascalvoc/coco/annotations/instances_val.json', 'r') as f:
    data = json.load(f)

# Update the 'category_id' for all entries in 'annotations'
for entry in data['annotations']:
    entry['category_id'] += 1  # Increment category_id by 1

# Update the 'id' for all entries in 'categories'
for category in data['categories']:
    category['id'] += 1  # Increment id by 1

# Save the modified JSON back to a file
with open('/home/bibahaduri/pascalvoc/coco/annotations/instances_u_val.json', 'w') as f:
    json.dump(data, f, indent=2)