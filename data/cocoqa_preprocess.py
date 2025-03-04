import os
import argparse
from http.client import responses

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

def download_coco_images(category: str) -> None:
    """
    Download COCO images based on super category according to link:
    https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    :param category:
    :return:
    """
    annotation_file = 'instances_train2014.json'
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
    output_dir = 'train2014'
    os.makedirs(output_dir, exist_ok=True)

    for img_id in img_ids[:10]:
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

def main(args) -> None:
    if args.download:
        print('Downloading COCO-QA dataset...')
        # download_cocoqa()
    if args.category:
        download_coco_images(args.category)


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
        default='vehicle',
        help='Category of images to download'
    )
    args = parser.parse_args()
    main(args)