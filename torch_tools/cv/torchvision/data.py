from torchvision import transforms as t

transforms_info = dict(
    tensor_to_image = [
        t.ToPilImage
    ],
    
    to_tensor= t.ToTensor,
    resize= t.Resize,
    crop= t.CenterCrop,
    normalize = t.Normalize,
    container_for_transforms = t.Compose,
)


def example_transforms():
    t.Compose([
        t.Resize(256),
        t.ToTensor()
    ])