import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1729], help="Range of seeds"
    )
    args = parser.parse_args()
    return args

def generate_img_ids():
    anno_ids = {i: {} for i in ID2CLASS.keys()}

    data_path = "/home/bibahaduri/pascalvoc/coco/annotations/instances_train.json"##"datasets/cocosplit/datasplit/trainvalno5k.json"
    data = json.load(open(data_path))
    file2id = {}
    for i in data["images"]:
        file2id[i["file_name"]] = i["id"]
    

    for j in ID2CLASS.keys():
        for i in [1,2,3,5,10]:
            anno_ids[j][i] = []
            filename = "/home/bibahaduri/pascalvoc/coco/annotations/splits/vocsplit/box_{}shot_{}_train.txt".format(i, ID2CLASS[j])
            with open(filename, 'r') as file:
                for line in file:
                    # Split the line to extract the file name
                    file_path = line.strip()
                    file_name = os.path.basename(file_path)
                    anno_ids[j][i].append(file2id[file_name])

    return anno_ids
            #open file box_Ishot_ID2class[j]_train.txt
            ###loop through lines 
                #get filename  and get associated image_id  put it in the anno_ids[j][i]

def generate_seeds(args):
    data_path = "/home/bibahaduri/pascalvoc/coco/annotations/instances_train.json"##"datasets/cocosplit/datasplit/trainvalno5k.json"
    data = json.load(open(data_path))
    anno_ids = generate_img_ids()
    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat)

    id2img = {}
    for i in data["images"]:
        id2img[i["id"]] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for i in args.seeds:##range([0], args.seeds[1]):)
        random.seed(i)
        for c in ID2CLASS.keys():
            # img_ids = {}
            # for a in anno[c]:
            #     if a["image_id"] in img_ids:
            #         img_ids[a["image_id"]].append(a)
            #     else:
            #         img_ids[a["image_id"]] = [a]

            # sample_shots = []
            # sample_imgs = []
            for shots in [1,3,5,10]:##[1, 2, 3, 5, 10, 30]:

                sample_shots = []
                sample_imgs = []
                img_ids ={}
                img_ids_others = {}
                for a in anno[c]:
                    
                    if a["image_id"] in anno_ids[c][shots]:
                        if a["image_id"] in img_ids:
                            img_ids[a["image_id"]].append(a)
                        else:
                            img_ids[a["image_id"]] = [a]
                    else:
                        if a["image_id"] in img_ids_others:
                            img_ids_others[a["image_id"]].append(a)
                        else:
                            img_ids_others[a["image_id"]] = [a]

                flag = False
                while True:
                    if flag == False:
                        imgs = list(img_ids.keys())#, shots #random.sample(...)
                        img_ids_holder = img_ids
                    else:
                        imgs = random.sample(list(img_ids_others.keys()), shots)#, shots #random.sample(...)
                        img_ids_holder = img_ids_others
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s["image_id"]:
                                skip = True
                                break

                        if skip:
                            continue
                        if len(img_ids_holder[img]) + len(sample_shots) > shots:
                            if flag == False and len(sample_shots) < shots:
                                sample_shots.append(img_ids_holder[img][0])
                                sample_imgs.append(id2img[img])
                                continue
                            else:
                                continue
                        sample_shots.extend(img_ids_holder[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    flag = True
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    "info": data["info"],
                    "licenses": data["licenses"],
                    "images": sample_imgs,
                    "annotations": sample_shots,
                }
                save_path = get_save_path_seeds(
                    data_path, ID2CLASS[c], shots, i
                )
                new_data["categories"] = new_all_cats
                with open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    save_dir = os.path.join("datasets", "cocosplit", "seed" + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":


    #=========================Pascal VOC CLASSES====================#

    ID2CLASS = {
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
    }

    #=========================DOTA CLASSES====================#
    
    # ID2CLASS = {
    #     1: "plane",
    #     2:  "ship",
    #     3: "storage-tank",
    #     4: "baseball-diamond",
    #     5: "tennis-court",
    #     6: "basketball-court",
    #     7: "ground-track-field",
    #     8: "harbor",
    #     9: "bridge",
    #     10: "small-vehicle",
    #     11: "large-vehicle",
    #     12: "roundabout",
    #     13: "swimming-pool",
    #     14: "helicopter",
    #     15:  "soccer-ball-field",
    #     16: "container-crane",
    # }
    

#=========================DIOR CLASSES====================#
    """
    ID2CLASS = {
        1: "Airplane ",
        2: "Airport ",
        3: "Baseball field ",
        4: "Basketball court ",
        5: "Bridge ",
        6: "Chimney ",
        7: "Dam ",
        8: "Expressway service area ",
        9: "Expressway toll station ",
        10: "Golf course",
        11: "Ground track field ",
        12: "Harbor ",
        13: "Overpass ",
        14: "Ship ",
        15: "Stadium ",
        16: "Storage tank ",
        17: "Tennis court ",
        18: "Train station ",
        19: "Vehicle ",
        20: "Wind mill",
    }
    """
   




    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)