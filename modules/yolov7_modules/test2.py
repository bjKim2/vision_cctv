import av
import cv2
import numpy as np

def check_imshow():
    # Check if environment supports image displays
    try:
        # assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        print('a')
        cv2.imshow('test', np.zeros((1, 1, 3)))
        print('a')
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False
    
if __name__ == '__main__':
    test = check_imshow()
    print('3')
    # cv2.imshow('test', np.ones((1, 1, 3)))
    # cv2.waitKey(2000)