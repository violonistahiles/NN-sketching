import torch
import torch.nn as nn
import cv2
import numpy as np
import os


def detect_face(face_cascade, img):

  '''
  Input parameters:
    face_cascade - opencv face detector
    img - original image
  Output parameters:
    coordinates of largest face rectangle (format: center_x, center_y, width, height)
  '''

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.1, 4)
  if (len(faces)==0):
    return None
  
  wh = 0
  idx = 0
  for i in range(0,len(faces)):
    (x, y, w, h) = faces[i]
    if (wh < w*h):
      idx = i
      wh = w*h

  return faces[idx]


def crop_face(img, img_size, face=None):

  '''
  Input parameters: 
    img - input image
    img_size - image size for network input
    face - coordinates of face rectangle (format: center_x, center_y, width, height)
  Output parameters:
    cropped_img - image of cropped face with saved aspect ratios in size (img_size, img_size)
  '''

  if (face is None): # If no face coordinates just reshape image with saving aspect ration

    scale = np.min([img_size/(img.shape[0]), img_size/(img.shape[1])])
    new_h, new_w = int((img.shape[0])*scale), int((img.shape[1])*scale)
    face = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cropped_img = np.full((img_size, img_size, 3), 255)
    start_h, end_h = (img_size-new_h)//2, (img_size-new_h)//2 + new_h
    start_w, end_w = (img_size-new_w)//2, (img_size-new_w)//2 + new_w
    cropped_img[start_h:end_h, start_w:end_w, :] = face

    return cropped_img
  
  # Face coordinates
  x, y, w, h = face

  # Cropping face from original image
  top = np.max([y - int(h*0.5), 0])
  bottom = np.min([y + h + int(h*0.3), img.shape[0]])
  left = np.max([x - int(w*0.4), 0])
  right = np.min([x + w + int(w*0.4), img.shape[1]])
  face = img[top:bottom, left:right, :]

  # Reshaping image with saveing sapect ratio
  scale = np.min([img_size/(bottom-top), img_size/(right-left)])
  new_h, new_w = int((bottom-top)*scale), int((right-left)*scale)
  face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

  # Paste image in (img_size, img_size)
  cropped_img = np.full((img_size, img_size, 3), 255)
  start_h, end_h = (img_size-new_h)//2, (img_size-new_h)//2 + new_h
  start_w, end_w = (img_size-new_w)//2, (img_size-new_w)//2 + new_w

  cropped_img[start_h:end_h, start_w:end_w, :] = face

  return cropped_img


def predict_img(model, device, face_cascade, filename, result_dir, img_size):

  '''
  Input parameters:
    model - network model
    device - cpu or gpu if available
    face_cascade - opencv face detector
    filename - path of image for predict
    result_dir - path where predicted image will be saved
    img_size - network input size
  Output parameters:
    img_dir - path where predicted image was saved
  '''

  # Reading image
  img = cv2.imread(filename, cv2.IMREAD_COLOR)

  # Extending channels if image is grayscale
  if len(img.shape) == 2:
    img = img[:,:,np.newaxis]
  if img.shape[2] == 1:
    img = np.stack([img, img, img], axis=1)

  # Image preparation for predict
  face = detect_face(face_cascade, img)
  cropped_img = crop_face(img, img_size, face=face)
  cropped_img = cv2.cvtColor(cropped_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
  cropped_img = cropped_img / 255
  cropped_img = cropped_img.transpose((2,0,1))[np.newaxis,:,:,:]
  img_for_torch = torch.from_numpy(cropped_img).type(torch.FloatTensor)
  # Predict
  predict = model(img_for_torch.to(device))[0]
  predict = predict.detach().cpu().numpy().squeeze().reshape((img_size, img_size, 1))
  
  # Creating filename for predict saving
  img_name = filename[::-1]
  point_index = img_name.index('.')
  slash_index = img_name.index('/')
  img_name = img_name[point_index+1:slash_index][::-1]
  img_name = img_name + '_predict.jpg'
  img_dir = os.path.join(result_dir, img_name)
  cv2.imwrite(img_dir, np.uint8(predict*255))

  return img_dir