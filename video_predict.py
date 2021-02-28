def draw_sketch(module_path, video_path, out_path):
    '''
    Input parameters:
      module_path - directory where sketch drawing module located
      img_path - video for predict directory, including video name
      out_path - directory for predict saving, excluding video name
    Output parameters:
      out_path - directory where video is saved (format: out_path + video_name + '_predict.avi')
    '''

    # Load standard modules
    import torch
    import torch.nn as nn
    import cv2
    import numpy as np
    import os
    import sys
    sys.path.append(module_path)

    # Load custom modules
    from model.u2net import REBNCONVp, _upsample_like, RSU7, RSU6, RSU5, RSU4, RSU4F, U2NET
    from utils.utils_video import detect_face, crop_face, predict_img

    img_size = 512  # Image size for net input

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading net and weights
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(module_path + 'SD_model_dict.pt', map_location=device))
    model.eval()
    model.to(device)

    # Simple opencv face detector
    face_cascade = cv2.CascadeClassifier(module_path + 'haarcascade_frontalface_default.xml')

    # Prediction
    img_dir = predict_img(model, device, face_cascade, video_path, out_path, img_size)

    return img_dir
