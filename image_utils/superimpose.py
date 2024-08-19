from PIL import Image

def superimpose_images(img1_path, img2_path, center_point):
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Calculate the scale factor to fit img2 into img1 for square foot
    max_width = img1.width
    max_height = img1.height
    width_ratio = max_width / img2.width
    height_ratio = max_height / img2.height
    scale_factor = min(width_ratio, height_ratio)

    # Scale down img2
    new_width = int(img2.width * scale_factor)
    new_height = int(img2.height * scale_factor)
    img2_resized = img2.resize((new_width, new_height))

    # Calculate the top-left corner coordinates for placing img2
    x = center_point[0] - new_width // 2
    y = center_point[1] - new_height // 2

    # Superimpose img2 on img1
    img1.paste(img2_resized, (x, y), img2_resized)

    return img1

# Example usage
img1_path = "C:\\Users\\quest\\Pictures\\Screenshots\\610.png"
img2_path = "D:\\Floor generator\\houseganpp\\dump\\610.png"
center_point = (460,210)  # Example center point, adjust as needed

result_image = superimpose_images(img1_path, img2_path, center_point)
result_image.show()