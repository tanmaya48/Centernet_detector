import os
import torch
import numpy as np
from dataset import image_transforms
import cv2

image_size = 512
hm_size = 128
conf = 0.3

def get_infered_boxes(model,image,device = 'cpu'):
    image = image_transforms(image) 
    image = image.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        center_hm,reg_hm = model(image.unsqueeze(0))
    center_hm = center_hm.to('cpu').numpy()[0][0]
    reg_hm = reg_hm.to('cpu').numpy()[0]
    coordinates = np.nonzero(center_hm)
    coordinates_list = list(zip(coordinates[0], coordinates[1]))
    boxes = []
    for center in coordinates_list:
        y,x = center
        if center_hm[y,x] < conf:
            continue
        w, h = reg_hm[0, y, x], reg_hm[1, y, x]
        x_image = x*image_size/hm_size
        y_image = y*image_size/hm_size
        w_image = w*image_size
        h_image = h*image_size
        x1 = int(x_image - w_image/2)
        y1 = int(y_image - h_image/2)
        x2 = int(x1 + w_image)
        y2 = int(y1 + h_image)
        boxes.append([x1,y1,x2,y2])
    hm = np.concatenate((np.expand_dims(center_hm, axis=0),reg_hm),axis = 0).transpose([1,2,0])*255
    return hm,boxes


if __name__ == "__main__":
    model = torch.load("entire_model.pt")
    image_dir = "../../yolo_dataset/images/val/"
    images = os.listdir(image_dir)
    for image_name in images:
        image_path = os.path.join(image_dir,image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image,(image_size,image_size))
        hm,boxes = get_infered_boxes(model,image,'cuda:0') 
        for box in boxes:
            x1,y1,x2,y2 = box
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imwrite(image_name,image)
        cv2.imwrite(f"hm_{image_name}",hm)

    

    


