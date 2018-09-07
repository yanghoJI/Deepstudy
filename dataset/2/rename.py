import os
import cv2


imsize = 600
sidx = 20000

imglist = os.listdir('./')
print(imglist)
for impath in imglist:
    if '.JPG' in impath:
        im = cv2.resize(cv2.imread(impath), (imsize, imsize))
        savename = '{}.png'.format(int(sidx))
        print(savename)
        cv2.imwrite(savename, im)
        if sidx % 30 == 0:
            cv2.imshow(savename, im)
        sidx += 1
    else:
        pass

cv2.waitKey(-1)



#print(imglist)
