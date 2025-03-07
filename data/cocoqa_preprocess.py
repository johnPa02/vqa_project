import os
import argparse
import json

from pycocotools.coco import COCO
import requests


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

def download_coco_images(category: str,) -> None:
    """
    Download COCO images based on super category according to link:
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    :param category:
    :return:
    """
    annotation_file = 'annotations_trainval2014/annotations/instances_train2014.json'
    output_dir = 'train2014'
    if not os.path.exists(annotation_file):
        raise FileNotFoundError('Annotation file not found')

    coco = COCO(annotation_file)
    # Get all category names in super category
    categories = coco.loadCats(coco.getCatIds())
    cat_ids = [cat['id'] for cat in categories if cat['supercategory'] == category]
    if not cat_ids:
        raise ValueError(f'Cannot find category {category} in COCO dataset')

    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=cat_id))
    print(f'Found {len(img_ids)} images for category {category}')
    os.makedirs(output_dir, exist_ok=True)

    for img_id in img_ids:
        try:
            img_info = coco.loadImgs(img_id)[0]
            img_url = img_info['coco_url']
            img_path = os.path.join(output_dir, img_info['file_name'])

            response = requests.get(img_url)
            with open(img_path, 'wb') as f:
                f.write(response.content)
            print(f'Successfully downloaded {img_path}')
        except Exception as e:
            print(f'Error while downloading image {img_id}: {e}')
    print(f'Download completed for category {category}, saved at {output_dir}')

def process_cocoqa(mode: str) -> None:
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

        if not os.path.exists(img_path):
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
    if args.download:
        print('Downloading COCO-QA dataset...')
        # download_cocoqa()
    if args.category:
        print(f'Downloading COCO images for category {args.category}...')
        download_coco_images(args.category)
    # process_cocoqa('train')
    # process_cocoqa('test')


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