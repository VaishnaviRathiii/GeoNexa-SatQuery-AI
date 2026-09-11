from PIL import Image


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image

def get_image_info(image: Image.Image) -> dict:
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode
    }
    