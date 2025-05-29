import os
import json
import base64
from typing import List, Dict, Any
from random import random, randint

def extract_products_by_barcodes(
    products: List[Dict[str, Any]],
    barcodes_to_extract: List[str],
    image_dir: str
) -> List[Dict[str, Any]]:
    """
    Return a list of products matching any barcode in barcodes_to_extract,
    with exactly one matched barcode and the corresponding image embedded in base64.
    """
    extracted: List[Dict[str, Any]] = []
    barcode_set = set(barcodes_to_extract)

    for prod in products:
        if not isinstance(prod, dict):
            continue
        
        codigos = prod.get("codigos_barras", [])
        if not isinstance(codigos, list) or not codigos:
            continue

        prod_barcodes = {
            str(barcode["codigo"]) for barcode in codigos
            if isinstance(barcode, dict) and "codigo" in barcode
        }

        matched = prod_barcodes & barcode_set
        if not matched:
            continue

        # Pick exactly one barcode from the matched set
        code = next(iter(matched))

        price = float(prod["valor_produto"])
        sale = (
            randint(int(round(price) * 0.05), int(round(price) * 0.8))
            if random() > 0.7 else 0.0
        )
        record = {
            "filename": None,
            "barcode": code,
            "product_name": prod["descricao"],
            "category": prod["departamento"],
            "subcategory": prod["secao"],
            "price": price,
            "on_sale": sale > 0,
            "sale_percentage": int(round(sale / price * 100)) if sale else 0,
            "sale_price": float(f"{(price - sale):.2f}") if sale else 0.0,
            "stock": randint(40, 300),
            "embeddings": None
        }

        # Try loading the image for this barcode
        for ext in (".jpg", ".jpeg"):
            img_path = os.path.join(image_dir, filename:= f"{code}{ext}")
            if os.path.isfile(img_path):
                record["filename"] = filename
                break
        extracted.append(record)

    return extracted


def extract_barcodes_from_images(directory: str) -> List[str]:
    """
    Scans a directory for .jpg/.jpeg files and extracts barcodes from filenames.
    """
    barcodes: List[str] = []
    for filename in os.listdir(directory):
        lower = filename.lower()
        if lower.endswith(('.jpg', '.jpeg')):
            barcode = os.path.splitext(filename)[0].strip()
            barcodes.append(barcode)
    return barcodes


if __name__ == "__main__":
    # 1) Load your full JSON
    with open("./cadastro_produtos.json", "r", encoding="utf-8") as f:
        all_products = json.load(f)
    products = all_products.get("produtos", [])

    image_directory = "./db-images"
    barcodes_to_extract = extract_barcodes_from_images(image_directory)
    # print("Barcodes to extract:", len(barcodes_to_extract), "| examples:", barcodes_to_extract[:3])

    filtered = extract_products_by_barcodes(products, barcodes_to_extract, image_directory)

    output_path = "filtered_products.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(filtered)} products into {output_path}")
