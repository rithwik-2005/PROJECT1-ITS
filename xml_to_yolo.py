import os
import shutil
import xml.etree.ElementTree as ET

# Original dataset folder
dataset_dir = "dataset"  # change to your folder
# New folder to store YOLO dataset
yolo_dataset_dir = "dataset_yolo"
images_yolo_dir = os.path.join(yolo_dataset_dir, "images")
labels_yolo_dir = os.path.join(yolo_dataset_dir, "labels")

# Create folders
os.makedirs(images_yolo_dir, exist_ok=True)
os.makedirs(labels_yolo_dir, exist_ok=True)

classes = ["license_plate"]

# Process all XML files
for file in os.listdir(dataset_dir):
    if not file.lower().endswith(".xml"):
        continue  # skip non-XML files

    xml_path = os.path.join(dataset_dir, file)
    print("Processing:", xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        cls_id = classes.index(cls_name) if cls_name in classes else 0

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    # Save YOLO label in labels folder
    txt_file = os.path.join(labels_yolo_dir, file.replace(".xml", ".txt"))
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))

    # Copy corresponding image to images_yolo_dir
    base_name = file.replace(".xml", ".jpg")
    src_img_path = os.path.join(dataset_dir, base_name)
    dst_img_path = os.path.join(images_yolo_dir, base_name)
    if os.path.exists(src_img_path):
        shutil.copy2(src_img_path, dst_img_path)
    else:
        print(f"Warning: Image {base_name} not found!")

print(f"Conversion complete! YOLO dataset stored in '{yolo_dataset_dir}/'")

