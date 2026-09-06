from PIL import Image

from ai.vlm import prepare_inputs


class MockProcessor:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "test prompt"

    def __call__(self, text, images, padding, return_tensors):
        return {
            "input_ids": [[1, 2, 3]],
            "pixel_values": [[0.0]]
        }


def test_prepare_inputs():
    processor = MockProcessor()
    image = Image.new("RGB", (512, 512))

    inputs = prepare_inputs(
        processor,
        image,
        "How many buildings are visible?"
    )

    assert "input_ids" in inputs
    assert "pixel_values" in inputs