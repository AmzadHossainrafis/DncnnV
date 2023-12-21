import albumentations as A 
t = A.Compose([ 
    A.Resize(256, 256), 
    A.HorizontalFlip(p=0.5), 
    A.RandomBrightnessContrast(p=1),
])




