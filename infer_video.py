import cv2
from predict import predict_img,mask_to_image
import logging
from unet import UNet
from torchvision import transforms
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np

from prettytable import PrettyTable


def main(video_source=0):
    # Open the video source
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    state_dict = torch.load(r"C:\Users\Soham\Desktop\Road Segmentation project\Pytorch-UNet\unet_carvana_scale0.5_epoch2.pth", map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            # Convert the frame from BGR to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        # Segment the frame
        mask_frame = predict_img(net=net,
                           full_img=pil_image,
                           scale_factor=0.5,
                           device=device)
        result = mask_to_image(mask_frame, mask_values)
        cv2_result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        # Display the original and segmented frames
        print(cv2_result.shape)
        print(frame.shape)
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Segmented Frame', cv2_result)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()