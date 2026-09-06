from PIL import Image
from ai.image_utils import get_image_info


def test_image_info():
    image = Image.new("RGB", (512, 512))
    info = get_image_info(image)

    assert info["width"] == 512
    assert info["height"] == 512
    assert info["mode"] == "RGB"
    