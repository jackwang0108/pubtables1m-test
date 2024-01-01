# Standard Library
import shutil
import xml.etree.ElementTree as ET

from os import PathLike
from pathlib import Path
from typing import List, Tuple

# Third-Party Library
import tqdm
from PIL import Image, ImageDraw


def get_example(base_dir: Path, image: Path) -> Tuple[Path]:
    xml = Path("train") / f"{image.stem}.xml"
    root_dir: Path = Path(__file__).resolve().parent
    (copy_image := root_dir / image).parent.mkdir(exist_ok=True)
    (copy_xml := root_dir / xml).parent.mkdir(exist_ok=True)
    if not copy_image.exists():
        shutil.copy(str(base_dir / image), str(root_dir / image))
    if not copy_xml.exists():
        shutil.copy(str(base_dir / xml), str(root_dir / xml))
    return copy_image, copy_xml


def get_filelist(root_dir: Path) -> List[str]:
    assert root_dir.exists(), f"{root_dir} doesn't exists"
    with (root_dir / "images_filelist.txt").open(mode="r") as f:
        return [line.strip() for line in f.readlines()]


def read_xml(xml_file: Path):
    assert xml_file.exists(), f"{xml_file} doesn't exists"
    with xml_file.open(mode="r") as f:
        tree: ET = ET.parse(f)

    root = ET.parse(tree)
    # 使用内置的深度优先遍历, iter返回一个迭代器
    # child: ET.Element
    # for child in root.iter():
    #     print(f"{child.tag=}")
    #     print(f"{child.attrib=}")
    #     print(f"{child.text=}")

    def traverse(element: ET.Element, depth=0):
        print("\t" * depth, f"{element.tag}")
        print("\t" * depth, f"{element.attrib=}")
        print("\t" * depth, f"{element.text=}")
        for child in element:
            traverse(child, depth + 1)
    traverse(root)


def get_class_map(data_type="detection"):
    if data_type == 'structure':
        class_map = {
            'table': 0,
            'table column': 1,
            'table row': 2,
            'table column header': 3,
            'table projected row header': 4,
            'table spanning cell': 5,
            'no object': 6
        }
    else:
        class_map = {'table': 0, 'table rotated': 1, 'no object': 2}
    return class_map


def read_pascal_voc(xml_file: Path) -> Tuple[List[float], List[str]]:
    assert xml_file.exists(), f"{xml_file} doesn't exists"
    # get tree and root
    with xml_file.open(mode="r") as f:
        tree: ET = ET.parse(f)
    root: ET.Element = tree.getroot()

    # get bounding boxs
    bboxs = []
    labels = []

    for _object in root.iter("object"):
        label = _object.find("name").text

        for box in _object.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        bbox = [xmin, ymin, xmax, ymax]

        bboxs.append(bbox)
        labels.append(label)

    return bboxs, labels


def draw_bboxs(image_path: Path, bboxs: List[List[float]], labels: List[str]):
    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)
    for bbox, label in zip(bboxs, labels):
        xmin, ymin, xmax, ymax = bbox

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red")

        draw.text((xmin, ymin - 10), label, fill="red")

    root_dir = Path(__file__).resolve().parent / 'bbox'
    root_dir.mkdir(exist_ok=True)
    image.save(root_dir / f"{image_path.stem}.jpg")


def get_all_labels(base_dir: Path):
    labels = set()
    base_dir = base_dir / "train"
    filelist = list(base_dir.iterdir())
    for file in tqdm.tqdm(filelist):
        with file.open(mode="r") as f:
            tree: ET = ET.parse(f)
        root: ET.Element = tree.getroot()
        for _object in root.iter("object"):
            labels.add(s := _object.find("name").text)
            if s == "table rotated":
                print(file)
    print(labels)


if __name__ == "__main__":
    detection_dir = Path(
        "/home/jack/Datasets/pubtables-1m/PubTables-1M-Detection").resolve()
    filelist = get_filelist(detection_dir)
    # for image in [Path(i) for i in filelist[:10]]:
    #     image, xml = get_example(detection_dir, image)
    #     bboxs, labels = read_pascal_voc(xml)
    #     draw_bboxs(image, bboxs, labels)
    get_all_labels(detection_dir)
