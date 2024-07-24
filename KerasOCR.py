import numpy as np
import torch
import comfy

import keras_ocr
import cv2

def calculate_area(box):
    box = np.array(box)
    return cv2.contourArea(box)

# Function to draw mask over detected text areas
def draw_masks(image, predictions):
    # Create a mask for the image
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Iterate over the predictions and draw rectangles on the mask
    for word, box in predictions:
        box = np.array(box).astype(int)
        cv2.fillPoly(mask, [box], (255, 255, 255))  # Filling the polygon with white color
    
    # Combine the original image with the mask
    masked_image = cv2.addWeighted(image, 0, mask, 1, 0)
    return masked_image

# Function to create a binary mask with white regions for detected text
def create_text_mask(image, predictions):
    # Create a black mask with the same size as the image
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Iterate over the predictions and draw white rectangles on the mask
    for word, box in predictions:
        box = np.array(box).astype(int)
        cv2.fillPoly(mask, [box], (255, 255, 255))  # Fill the polygon with white color
    
    return mask

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask

class KerasOCR:
    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
            "hidden": {"prompt": "PROMPT"}
        }

        
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "keras_ocr"

    CATEGORY = "PerfectDiffusion"

    def keras_ocr(self, images, prompt = None):
        
        if images is not None:
            images_to_ocr = []
            for (batch_number, image) in enumerate(images):
                i = 255. * image.cpu().numpy()
                cv_img = np.clip(i, 0, 255).astype(np.uint8)
                cv_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                images_to_ocr.append(cv_image)
                
            prediction_groups = self.pipeline.recognize(images_to_ocr)
            # fig, axs = plt.subplots(nrows=len(images),ncols=2, figsize=(20, 20))
            masks = []
            for i, (image, predictions) in enumerate(zip(images, prediction_groups)):
                tmask = create_text_mask(image, predictions)
                float_mask = np.array(tmask).astype(np.float32) / 255.0
                tmask_np = torch.from_numpy(float_mask)[None,]
                tmask_np_mask = tmask_np[:, : , :, 0]
                masks.append(tmask_np_mask.unsqueeze(0))


            outmasks = None
            if len(masks) == 1:
                # outmasks = make_3d_mask(masks[0])
                outmasks = masks[0]

            elif len(masks) > 1:
                # mask1 = make_3d_mask(masks[0])
                mask1 = masks[0]
                for mask2 in masks[1:]:
                    # mask2 = make_3d_mask(mask2)
                    if mask1.shape[1:] != mask2.shape[1:]:
                        mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
                    mask1 = torch.cat((mask1, mask2), dim=0)
                outmasks = mask1
                
            return (outmasks,)

# plt.show()
NODE_CLASS_MAPPINGS = {
    "KerasOCR" : KerasOCR
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KerasOCR" : "Keras OCR mask"

}