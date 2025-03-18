import base64
import io
import mimetypes
from PIL import Image
def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"

def decode_image(encoded_string: str, save_path: str):
    """Decodes an image from a base64 string."""
    encoded_string = encoded_string.split(',')[1]
    with open(save_path, "wb") as image_file:
        image_file.write(base64.b64decode(encoded_string.encode('utf-8')))

def sort_scores_dict(scores_dict):
    return dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=1))