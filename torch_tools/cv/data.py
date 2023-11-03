from PIL import Image
import requests
from io import BytesIO

example_pic = 'https://static01.nyt.com/images/2019/04/28/world/28london-marathon1/28london-marathon1-superJumbo.jpg'

def image_from_url(
    url=example_pic,
    ):
    """
    to get an example image 
    1) google an image topic
    2) right click image > copy image address
    """
    response = requests.get(url)
    return Image.open(BytesIO(response.content))