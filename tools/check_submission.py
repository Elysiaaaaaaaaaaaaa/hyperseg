from pathlib import Path
import sys
from PIL import Image

def check(pred_dir, image_dir, classes=9):
    pred, images = Path(pred_dir), {p.name for p in Path(image_dir).glob('*.png')}; files=list(pred.glob('*.png'))
    assert {p.name for p in files} == images, 'filename set mismatch'
    for p in files:
        im=Image.open(p); assert im.mode == 'L' and im.size == (1024,1024), f'invalid {p.name}'
        ids=set(im.getdata()); assert ids <= set(range(classes)), f'invalid class id in {p.name}'
    print(f'OK: {len(files)} predictions')
if __name__ == '__main__': check(sys.argv[1], sys.argv[2])
