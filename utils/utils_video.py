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
    if (len(faces) == 0):
        return None

    wh = 0
    idx = 0
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        if (wh < w * h):
            idx = i
            wh = w * h

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

    # Face coordinates
    x, y, w, h = face

    # Cropping face from original image
    top = np.max([y - int(h * 0.5), 0])
    bottom = np.min([y + h + int(h * 0.3), img.shape[0]])
    left = np.max([x - int(w * 0.5), 0])
    right = np.min([x + w + int(w * 0.5), img.shape[1]])
    face = img[top:bottom, left:right, :]

    # Reshaping image with saveing aspect ratio
    scale = np.min([img_size / (bottom - top), img_size / (right - left)])
    new_h, new_w = int((bottom - top) * scale), int((right - left) * scale)
    face = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Paste image in (img_size, img_size)
    cropped_img = np.full((img_size, img_size, 3), 255)
    start_h, end_h = (img_size - new_h) // 2, (img_size - new_h) // 2 + new_h
    start_w, end_w = (img_size - new_w) // 2, (img_size - new_w) // 2 + new_w

    cropped_img[start_h:end_h, start_w:end_w, :] = face

    return cropped_img


def predict_img(model, device, face_cascade, filename, result_dir, img_size):
    '''
    Input parameters:
      model - network model
      device - cpu or gpu if available
      face_cascade - opencv face detector
      filename - path of video for predict
      result_dir - path where predicted video will be saved
      img_size - network input size
    Output parameters:
      img_dir - path where predicted video was saved
    '''

    # Creating filename for predict saving
    video_dir = result_dir + filename[filename.rindex('/') + 1:filename.rindex('.')] + '_predict.avi'

    # Setting video reader and writter
    video_reader = cv2.VideoCapture(filename)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(video_dir,
                                   cv2.VideoWriter_fourcc(*'MPEG'),
                                   20.0,
                                   (512, 512))

    wh_history = []

    for i in range(nb_frames):
        _, img = video_reader.read()

        # Extending channels if image is grayscale
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        if img.shape[2] == 1:
            img = np.stack([img, img, img], axis=1)

        # Image preparation for predict
        face = detect_face(face_cascade, img)

        # Some filters for stabilization
        if i == 0:
            if face is None:
                face = 0, 0, frame_w, frame_h
            # Will take first face rectangle size as constant size for cropping face
            w_et, h_et = face[2], face[3]

        if face is None:
            face = face_prev
        else:
            wh = face[2] * face[3]
            if i != 0:  # If detected face is relatively small, it is more probably error face, so take good previous
                if wh < (np.mean(wh_history) / 2):
                    face = face_prev
                    wh = face[2] * face[3]
            # correcting face coordinates to w_et, h_et
            x_center, y_center = face[0] + face[2] // 2, face[1] + face[3] // 2
            x = np.max(x_center - w_et // 2, 0)
            y = np.max(y_center - h_et // 2, 0)
            face = x, y, w_et, h_et

        cropped_img = crop_face(img, img_size, face=face)
        cropped_img = cv2.cvtColor(cropped_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cropped_img = cropped_img / 255
        cropped_img = cropped_img.transpose((2, 0, 1))[np.newaxis, :, :, :]
        img_for_torch = torch.from_numpy(cropped_img).type(torch.FloatTensor)
        # Predict
        predict = model(img_for_torch.to(device))[0]
        predict = predict.detach().cpu().numpy()

        frame = predict.squeeze()
        frame = np.stack([frame, frame, frame], axis=-1)

        video_writer.write((frame * 255).astype(np.uint8))

        face_prev = face
        wh_history.append(wh)

    video_writer.release()
    video_reader.release()

    return video_dir
