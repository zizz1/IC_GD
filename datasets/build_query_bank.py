# build_query_bank.py

import torch
import os
import random
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# 这是一个通用的做法，将你的项目根目录添加到Python路径中
# 以便我们可以导入项目中的模块，比如 datasets.coco
# 如果你的项目结构不同，你可能需要调整这个路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.coco import build as build_coco_dataset
from datasets.transforms import make_coco_transforms
from util.slconfig import SLConfig # 假设你的配置是这样加载的

def get_args_parser():
    parser = argparse.ArgumentParser('Build Query Bank', add_help=False)
    # --- 你需要根据你的项目，在这里提供正确的配置文件路径 ---
    parser.add_argument('--config_file', '-c', type=str, required=True, help="Path to the dataset configuration file.")
    parser.add_argument('--num_queries', type=int, default=10, help="Number of query examples to save per class.")
    parser.add_argument('--output_path', type=str, default="query_bank.pth", help="Path to save the output query bank file.")
    parser.add_argument('--min_box_size', type=int, default=32, help="Minimum size (width or height) of a bounding box to be considered a valid query.")
    return parser

def main(args):
    # 加载你的项目配置
    cfg = SLConfig.fromfile(args.config_file)

    print("Step 1: Loading dataset...")
    # 我们需要加载原始图片，所以不使用任何变换
    dataset = build_coco_dataset(image_set='train', args=cfg, datasetinfo=cfg.dataset.train)
    # 我们还需要一个带变换的实例，用来处理裁切出的图片
    transform, _ = make_coco_transforms(image_set='train', args=cfg)
    print("Dataset loaded successfully.")

    print("\nStep 2: Collecting all valid annotations by class...")
    all_annotations_by_class = {}
    for cat_id, cat_info in dataset.coco.cats.items():
        class_name = cat_info['name']
        
        ann_ids = dataset.coco.getAnnIds(catIds=[cat_id])
        annotations = dataset.coco.loadAnns(ann_ids)
        
        valid_annos = []
        for ann in annotations:
            # 筛选掉太小的或不合格的标注
            if ann['iscrowd'] == 0 and ann['bbox'][2] >= args.min_box_size and ann['bbox'][3] >= args.min_box_size:
                valid_annos.append({
                    'image_id': ann['image_id'],
                    'bbox': ann['bbox']
                })
        all_annotations_by_class[class_name] = valid_annos
    
    print("Annotation collection complete.")
    for name, annos in all_annotations_by_class.items():
        print(f"  - Found {len(annos)} valid instances for class '{name}'")


    print(f"\nStep 3: Building query bank with {args.num_queries} queries per class...")
    query_bank = {}
    # 使用 tqdm 创建一个漂亮的进度条
    for class_name, all_annos in tqdm(all_annotations_by_class.items()):
        if not all_annos:
            continue

        num_to_select = min(len(all_annos), args.num_queries)
        selected_annos = random.sample(all_annos, num_to_select)

        query_bank[class_name] = []
        for item in selected_annos:
            img_info = dataset.coco.loadImgs(item['image_id'])[0]
            img_path = os.path.join(dataset.root, img_info['file_name'])
            img = Image.open(img_path).convert('RGB')
            
            x, y, w, h = item['bbox']
            # PIL crop's box is (left, upper, right, lower)
            crop_box = (x, y, x + w, y + h)
            query_img = img.crop(crop_box)
            
            # 应用变换，转换成Tensor
            query_tensor, _ = transform(query_img, None)
            query_bank[class_name].append(query_tensor)

    print("Query bank built successfully.")

    print(f"\nStep 4: Saving query bank to {args.output_path}...")
    torch.save(query_bank, args.output_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Query Bank Builder', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)