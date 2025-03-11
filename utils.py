import os
import json

DIR = '../data/train2014/'
def remove_image(filtered_file: str) -> None:
    """
    Remove image that are not in the filtered file from the image directory
    :param
    filtered_file: metadata file with filtered images
    img_root: directory containing images
    :return:
    """
    with open(filtered_file, 'r') as f:
        data = json.load(f)
        img_paths = [d['img_path'].replace('train2014/','') for d in data]
        ques_ids = [d['ques_id'] for d in data]
    img_files = os.listdir(DIR)
    uq_image_files = list(set(img_files))
    uq_ques_ids = list(set(ques_ids))
    for img_file in img_files:
        if img_file not in img_paths:
            os.remove(os.path.join(DIR, img_file))
    print(f"Removed images not in {filtered_file} from {DIR}")

if __name__ == '__main__':
    remove_image('../data/filtered_data_train.json')