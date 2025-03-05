import cv2
import numpy as np

hm_size = 128
image_size = 512


def draw_msra_gaussian(heatmap, center, sigma=2):
    tmp_size = sigma * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
      heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
      g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def make_hm_regr(targets):
    # make output heatmap for single class
    hm = np.zeros([hm_size, hm_size])
    # make regr heatmap 
    regr = np.zeros([2, hm_size, hm_size])
    if len(targets) == 0:
        return hm, regr
    """
    made change here
    target = n,4 array of normalized xc,yc,w,h (YOLO format)
    """
    for target in targets:
        xc,yc,w,h = target
        hm = draw_msra_gaussian(hm, [int(xc*hm_size), int(yc*hm_size)], 
                                #sigma=np.clip(c[2]*c[3]//2000, 2, 4))    
                                sigma=np.clip(w*h*image_size*image_size//2000, 2, 4))    

    # convert targets to its center.
    #regrs = center[:, 2:]/input_size/IN_SCALE
    """
    here they had normalized the width and height values, not needed since I have assumed YOLO format
    also creating regr map in a separate loop as that was done in the reference code, it can be done in the above loop as well
    """
    # plot regr values to mask
    for target in targets:
        xc,yc,w,h = target
        grid_x,grid_y = int(xc*hm_size), int(yc*hm_size)
        for i in range(-2, 3):
            for j in range(-2, 3):
                try:
                    regr[0, grid_y+i, grid_x+j] = w 
                    regr[1, grid_y+i, grid_x+j] = h 
                except:
                    pass
    """
    here, they had transposed the matrix since they were indexing x,y... I have done it as y,x since it is chw
    so there is no need for transposing 
    """
    #regr[0] = regr[0].T; regr[1] = regr[1].T;
    return hm, regr