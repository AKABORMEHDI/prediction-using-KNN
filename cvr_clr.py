from PIL import Image


image_path = "71.jpeg"

original_image = Image.open(image_path)

resized_image = original_image.resize((8, 8))

bw_image = resized_image.convert("L")

inverted_image = Image.eval(bw_image, lambda x: 255 - x)

inverted_image.save("image_pretraite.jpg")

inverted_image.show()
