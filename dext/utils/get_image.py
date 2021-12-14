from paz.processors.image import LoadImage


def get_image(raw_image_path):
    loader = LoadImage()
    raw_image = loader(raw_image_path)
    raw_image = raw_image.astype('uint8')
    return raw_image
