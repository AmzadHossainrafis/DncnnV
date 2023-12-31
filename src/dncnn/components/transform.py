import albumentations as A

t = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1),
    ]
)


t2 = A.Compose(
    [
       # reduce the brightness of images to make low light images
        A.GaussianBlur(p=1, blur_limit=(3, 7)),
    ]
)
