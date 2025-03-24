import os
import argparse
import json

from pycocotools.coco import COCO
import requests

ANN_FILE = 'annotations_trainval2014/annotations/instances_val2014.json'
IMAGE_DOWNLOAD_DIR = 'val2014'

def download_cocoqa() -> None:
    """
    Download COCO-QA dataset, including questions, answers and image_ids
    :return:
    """
    if os.path.exists('coco_qa'):
        raise FileExistsError('COCO-QA dataset already exists')
    try:
        os.system('wget http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip -P zip/')
        os.system('unzip zip/cocoqa-2015-05-17.zip -d coco_qa/')
    except Exception as e:
        print('Error while downloading COCO-QA dataset: %s' % e)

def get_images_by_category(category: str, nums: int) -> list[int]:
    """
    Get all images from COCO dataset based on super category
    :param
    category: super category of images
    nums: number of images to retain
    :return:
    """
    if not os.path.exists(ANN_FILE):
        raise FileNotFoundError('Annotation file not found')

    coco = COCO(ANN_FILE)
    # Get all category names in super category
    categories = coco.loadCats(coco.getCatIds())
    cat_ids = [cat['id'] for cat in categories if cat['supercategory'] == category]
    if not cat_ids:
        raise ValueError(f'Cannot find category {category} in COCO dataset')

    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=cat_id))
    print(f'Found {len(img_ids)} images for category {category}')
    return img_ids[:nums]

def download_coco_images(img_ids: list[int]) -> None:
    os.makedirs(IMAGE_DOWNLOAD_DIR, exist_ok=True)
    coco = COCO(ANN_FILE)
    for img_id in img_ids:
        try:
            img_info = coco.loadImgs(img_id)[0]
            img_url = img_info['coco_url']
            img_path = os.path.join(IMAGE_DOWNLOAD_DIR, img_info['file_name'])
            if os.path.exists(img_path):
                print(f'{img_path} already exists, skipping download')
                continue

            response = requests.get(img_url)
            with open(img_path, 'wb') as f:
                f.write(response.content)
            print(f'Successfully downloaded {img_path}')
        except Exception as e:
            print(f'Error while downloading image {img_id}: {e}')
    print(f'Download completed, saved at {IMAGE_DOWNLOAD_DIR}')

def process_cocoqa(mode: str, cat_img_ids: list[int]) -> None:
    """
    :param mode: 'train' or 'test'
    Process COCO-QA dataset to create train and test metadata files
    :return:
    """
    questions_file = f"coco_qa/{mode}/questions.txt"
    answers_file = f"coco_qa/{mode}/answers.txt"
    img_ids_file = f"coco_qa/{mode}/img_ids.txt"
    types_file = f"coco_qa/{mode}/types.txt"
    output_file = f"cocoqa_raw_{mode}.json"

    with open(questions_file, "r") as qf, \
            open(answers_file, "r") as af, \
            open(img_ids_file, "r") as iif, \
            open(types_file, "r") as tf:
        questions = qf.readlines()
        answers = af.readlines()
        img_ids = iif.readlines()
        types = tf.readlines()

        questions = [line.rstrip('\n') for line in questions]
        answers = [line.rstrip('\n') for line in answers]
        img_ids = [line.rstrip('\n') for line in img_ids]
        types = [line.rstrip('\n') for line in types]

        assert len(questions) == len(answers) == len(img_ids) == len(types), \
            "Length of questions, answers, img_ids and types should be the same"

    subtype = 'train2014' if mode == 'train' else 'val2014'
    cocoqa_data = []
    for idx, (question, answer, img_id, type) in enumerate(zip(questions, answers, img_ids, types)):
        img_path = f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"

        if int(img_id) not in cat_img_ids:
            continue

        question = f"{question.strip()} ?"
        data_entry = {
            "ques_id": idx,
            "question": question,
            "img_path": img_path,
            "ans": answer,
            "type": type
        }
        cocoqa_data.append(data_entry)

    with open(output_file, 'w') as f:
        json.dump(cocoqa_data, f)
    print(f"Processed {mode} data saved at {output_file}")

def main(args) -> None:
    # preprocess pipeline
    # 1. download cocoqa dataset
    if args.download:
        print('Downloading COCO-QA dataset...')
        # download_cocoqa()
    # 2. get vehicle images from COCO dataset
    # img_ids = get_images_by_category(args.category, 3550)
    # 3. create metadata files for train and test
    # process_cocoqa('test', img_ids)
    # 4. classify questions involving vehicles (run OLLAMA_SERVER notebook)
    # 5. download images
    with open('vehicle_raw_test.json', 'r') as f:
        data = json.load(f)
        img_ids = [int(d['img_path'].split('_')[-1].replace('.jpg', '')) for d in data]
    # download_coco_images()
    img_ids = list(set(img_ids))
    print(f'Number of unique images: {len(img_ids)}')
    download_coco_images(img_ids)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download COCO-QA dataset'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Category of images to download'
    )
    args = parser.parse_args()
    main(args)