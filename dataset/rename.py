import os
import cv2


imsize = 600
sidx = 0

imglist = os.listdir('./')
for impath in imglist:
    if '.jpg' in impath:
        im = cv2.resize(cv2.imread(impath), (imsize, imsize))
        savename = '{}.png'.format(int(sidx))
        cv2.imwrite(savename, im)
        if sidx % 30 == 0:
            cv2.imshow(savename, im)
        sidx += 1
    else:
        pass

cv2.waitKey(-1)



#print(imglist)
