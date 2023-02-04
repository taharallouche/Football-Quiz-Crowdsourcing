from PIL import Image, ImageEnhance
import os
import PIL


if __name__ == "__main__":
    # Resize and diminish saturation of all images in given path
    path = ""  # Path to the folder containing the images to resize
    images = [
        file for file in os.listdir(path) if file.endswith(("jpeg", "png", "jpg"))
    ]  # Add your new format if not included
    for image in images:
        img = Image.open(path + "/" + image)
        converter = PIL.ImageEnhance.Color(img)
        img2 = converter.enhance(0.18)  # Diminish color saturation
        img2.thumbnail((350, 350))  # Resize the image
        img2.save("resized_" + image, optimize=True, quality=40)
