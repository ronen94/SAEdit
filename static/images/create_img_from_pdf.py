from pdf2image import convert_from_path
from rembg import remove
from PIL import Image
import io

# Extract images from PDF at high resolution
pdf_path = '"C:/Users/ronen/Downloads/Ronen_Sparse_AE (5).pdf"'
images = convert_from_path(pdf_path, dpi=300)  # Adjust DPI as needed (300-600 for high res)

# Process each page/image
for i, image in enumerate(images):
    # Remove background
    output = remove(image)

    # Save as PNG (preserves transparency)
    output.save(f'image_{i}_no_bg.png', 'PNG')
    print(f'Saved: image_{i}_no_bg.png')