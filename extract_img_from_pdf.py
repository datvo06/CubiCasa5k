
from pdf2image import convert_from_bytes
import os
import sys
import io

def pdf_to_images_v2(pdf_path, output_folder, dpi=300):
    # Read the PDF file as bytes
    with open(pdf_path, 'rb') as pdf_file:
        pdf_bytes = pdf_file.read()
    
    page_index = 0
    while True:
        try:
            # Convert the current page to an image
            images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt='png', first_page=page_index+1, last_page=page_index+1)
            if not images:
                break
            
            image = images[0]
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Save the image to the output folder
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_index+1}.png")
            with open(output_file, 'wb') as img_file:
                img_file.write(img_byte_arr)
            
            print(f"Saved image {output_file}")
            page_index += 1
        except IndexError:
            break

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_images.py <pdf_path> <output_folder>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_to_images_v2(pdf_path, output_folder)

