from PIL import Image, ImageDraw, ImageFont
import os


def create_comparison_image(image_paths, arrow_label, output_path='comparison.png'):
    """
    Creates a side-by-side comparison of 4 images with labels and arrows.

    Args:
        image_paths: List of 4 image file paths
        arrow_label: Single label that appears above the arrows (e.g., "Editing Results")
        output_path: Path to save the output image
    """

    # Load images
    images = [Image.open(path).convert('RGB') for path in image_paths]

    # Resize all images to same size
    target_width = 300
    target_height = 300
    resized_images = [img.resize((target_width, target_height), Image.LANCZOS) for img in images]

    # Calculate dimensions
    spacing = 15  # Space between images
    top_margin = 60  # Space for top labels
    bottom_margin = 20  # Reduced bottom margin since no labels
    side_padding = 30

    # Calculate canvas size
    canvas_width = (4 * target_width) + (3 * spacing) + (2 * side_padding)
    canvas_height = target_height + top_margin + bottom_margin

    # Create canvas with white background
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)

    # Try to load fonts - prioritize more artistic/elegant fonts
    try:
        # Try Georgia for a more elegant serif look
        title_font = ImageFont.truetype("georgia.ttf", 22)
    except:
        try:
            # Try Times New Roman as fallback
            title_font = ImageFont.truetype("times.ttf", 22)
        except:
            try:
                # Try Book Antiqua
                title_font = ImageFont.truetype("BKANT.TTF", 22)
            except:
                try:
                    # Linux fonts
                    title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                                                    22)
                except:
                    # Final fallback
                    title_font = ImageFont.load_default()

    # Draw "Original Image" label above first image
    text = "Original Image"
    bbox = draw.textbbox((0, 0), text, font=title_font)
    text_width = bbox[2] - bbox[0]
    text_x = side_padding + (target_width - text_width) // 2
    text_y = 20
    draw.text((text_x, text_y), text, fill='black', font=title_font)

    # Calculate arrow region (from second image to last image)
    arrow_start_x = side_padding + target_width + spacing
    arrow_end_x = canvas_width - side_padding
    arrow_region_width = arrow_end_x - arrow_start_x

    # Draw arrow label centered above where the arrow will be
    bbox = draw.textbbox((0, 0), arrow_label, font=title_font)
    text_width = bbox[2] - bbox[0]
    label_x = arrow_start_x + (arrow_region_width - text_width) // 2
    label_y = 10
    draw.text((label_x, label_y), arrow_label, fill='black', font=title_font)

    # Draw single horizontal arrow line below the label
    line_y = 35
    draw.line([arrow_start_x, line_y, arrow_end_x, line_y], fill='black', width=2)

    # Draw single arrowhead at the end
    arrow_size = 10
    draw.polygon([
        (arrow_end_x, line_y),
        (arrow_end_x - arrow_size, line_y - arrow_size),
        (arrow_end_x - arrow_size, line_y + arrow_size)
    ], fill='black')

    # Paste images
    current_x = side_padding
    for idx, img in enumerate(resized_images):
        img_y = top_margin
        canvas.paste(img, (current_x, img_y))
        current_x += target_width + spacing

    # Save the result
    canvas.save(output_path, quality=95)
    print(f"Comparison image saved to: {output_path}")
    return canvas


# Example usage
if __name__ == "__main__":
    # Replace these with your actual image paths
    image_paths = [
        "C:/Users/ronen/projects/token_sliders/objects/rusty_sword/51/res_scale_0.0_seed51.png",
        "C:/Users/ronen/projects/token_sliders/objects/rusty_sword/51/res_scale_5.0_seed51.png",
        "C:/Users/ronen/projects/token_sliders/objects/rusty_sword/51/res_scale_7.0_seed51.png",
        "C:/Users/ronen/projects/token_sliders/objects/rusty_sword/51/res_scale_13.0_seed51.png",



    ]

    # Label for the arrow
    arrow_label = 'rusty'

    # Create the comparison image
    create_comparison_image(image_paths, arrow_label, '../static/images/editing_results/Slide7.JPG')