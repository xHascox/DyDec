import yolov5
import torch
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision


OBJECT_CATEGORY_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']  # These are the classes that Yolov5 can detect

def extract_predictions(predictions_, conf_thresh=0.5, n_boxes=5, classes=[], DEBUG=False):
    """
    Filters the predictions obtained from a model by their confidence and classes

    predictions_: The predictions from a model
    conf_thresh: Min. score for predictions to be included (float)
    n_boxes: Max. number of prediction to keep (takes the ones with highest score) (int)
    classes: classes to keep (all if empty) (list)

    Returns a tuple of (classes, boxes, scores) of the detections
    """
    # Filter by confidence threshold
    boxes = predictions_["boxes"]
    scores = predictions_["scores"]
    labels = predictions_["labels"]

    if DEBUG: print(f">>>> filtering {len(scores)} with threshold {conf_thresh}")
    filtered_indices = [i for i, score in enumerate(scores) if score >= conf_thresh]
    if len(filtered_indices) < 1:
        return [], [], []
    boxes = [boxes[i] for i in filtered_indices]
    scores = [scores[i] for i in filtered_indices]
    labels = [labels[i] for i in filtered_indices]

    # Apply Non-Maximum Suppression (NMS)
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    if DEBUG: print(f">>>> running NMS with {len(scores)} scores")
    keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.5).tolist()
    
    predictions_class = [OBJECT_CATEGORY_NAMES[i] for i in labels]  # For each prediction, get the predicted class
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in boxes]  # Get the predicted bounding boxes
    predictions_score = scores  # Get the prediction score

    predictions_class = [predictions_class[i] for i in keep_indices]
    predictions_boxes = [predictions_boxes[i] for i in keep_indices]
    predictions_score = [predictions_score[i] for i in keep_indices]

    predictions_t = sorted(range(len(predictions_score)), key=lambda i: predictions_score[i], reverse=True)  # Sort the prediction indices by their score

    # Put the other lists in the same order, by descending prediction score
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    if classes:  # only keep the n_boxes best detections of the specified classes, if they are above the confidence threshold
        predictions_boxes_n = [e for e, c, s in zip(predictions_boxes, predictions_class, predictions_scores) if c in classes and s >= conf_thresh][:n_boxes]
        predictions_class_n = [e for e, s in zip(predictions_class, predictions_scores) if e in classes and s >= conf_thresh][:n_boxes]
        predictions_scores_n = [e for e, c in zip(predictions_scores, predictions_class) if c in classes and e >= conf_thresh][:n_boxes]
    else:  # only keep the n_boxes best detections
        predictions_boxes_n = predictions_boxes[:n_boxes]
        predictions_class_n = predictions_class[:n_boxes]
        predictions_scores_n = predictions_scores[:n_boxes]

    if DEBUG: print(f"predictions_class_n {predictions_class_n}")
    if DEBUG: print(f"predictions_boxes_n {predictions_boxes_n}")
    if DEBUG: print(f"predictions_scores_n {predictions_scores_n}")
    return predictions_class_n, predictions_boxes_n, predictions_scores_n

def plot_image_with_boxes(to_delete_directory, img, path="patch.png", boxes=[], pred_cls=[], title="", scores=[], color=(0,255,0), colordict={}):
    """
    Visualizes an image with the detected object boxes

    img: Image to plot
    boxes: Prediction boxes (from extract_predictions)
    pred_cls: Class (from extract_predictions)
    title: The plot title (str)
    scores: Score (from extract_predictions)
    color: Default color value for the boxes (tuple)
    colordict: Dictionary with box colors for some classes (dict) like {"car":(255,0,0)}
    """
    text_size = 1
    text_th = 3
    rect_th = 2

    for i in range(len(boxes)):
        print(type(img), img.shape)
        print("==", (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      colordict.get(pred_cls[i], color))
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
        color=colordict.get(pred_cls[i], color), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img, pred_cls[i]+" "+f"{scores[i]:.2f}", (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    colordict.get(pred_cls[i], color), thickness=text_th)


    base, ext = os.path.splitext(os.path.basename(path))
    new_filename = os.path.join(to_delete_directory, f"{base}_v{ext}")
    
    plt.imsave(new_filename, img / 255, format='png')



class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.hyp = {'box': 0.05,
                        'obj': 1.0,
                        'cls': 0.5,
                        'anchor_t': 4.0,
                        'cls_pw': 1.0,
                        'obj_pw': 1.0,
                        'fl_gamma': 0.0
                        }

    def forward(self, x, targets=None):
        if self.training:
            print("self.training is true")
            outputs = self.model.model.model(x)
            print("Not Implemented ComputeLoss")
            return None

        else:
            print(f"self.training is false {x.shape}")
            outputs = self.model(x)
            return outputs



def does_pred_contain_class(scores, labels, target_label, threshold, k=2):
    n = 0
    for score, label in zip(scores, labels):
        # Check if the current label matches L and the score is greater than or equal to S
        if label in target_label:
            print(f">>> found {target_label} {score}")
        if label in target_label and score >= threshold:
            n = n + 1
    if n >= k:
        return True
    return False

def initialize_model_and_detector(INPUT_SHAPE):
    model = yolov5.load('yolov5s.pt')
    model = Yolo(model)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    detector = PyTorchYolo(model=model,
                        device_type=device,
                        input_shape=INPUT_SHAPE,
                        clip_values=(0, 255), 
                        attack_losses=("loss_total", "loss_cls",
                                        "loss_box",
                                        "loss_obj"))
    
    return detector







def contains_pedestrian(image, path, to_delete_directory):
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
    else:
        print("No GPU found.")

    image_longest_side =  max(image.height, image.width)
    NUMBER_CHANNELS = 3
    INPUT_SHAPE = (NUMBER_CHANNELS, image_longest_side, image_longest_side)

    variable_name = "detector"
    if not variable_name in globals():
        global detector
        detector = initialize_model_and_detector(INPUT_SHAPE)
    
    from torchvision.transforms import transforms
    img_height = min(image.height, image_longest_side)
    padding_top = (INPUT_SHAPE[1] - img_height) // 2
    padding_bottom = INPUT_SHAPE[1] - img_height - padding_top
    transform = transforms.Compose([
            transforms.Pad((0, padding_top, 0, padding_bottom)),
            transforms.Resize((INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=transforms.InterpolationMode.BICUBIC),            
            transforms.ToTensor()
        ])
    im = transform(image).numpy()

    tensor = torch.tensor([im[:3, :, :]*255]).to("cpu")#.unsqueeze(0)

    detections = detector.predict(tensor)

    predictions = extract_predictions(detections[0])
    
    plot_image_with_boxes(to_delete_directory, np.ascontiguousarray(tensor[0].numpy().transpose(1,2,0).astype(np.float32)), path, predictions[1], predictions[0], "visualization", predictions[2], colordict={"person":(255,0,255)})
    #https://answers.opencv.org/question/216152/typeerror-expected-cvumat-for-argument-img/

    contains = does_pred_contain_class(predictions[2], predictions[0], target_label=["person"], threshold=0.1)

    return contains
