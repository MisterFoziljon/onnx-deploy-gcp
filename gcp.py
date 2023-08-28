import streamlit as st
import cv2
import tempfile
import onnxruntime
import numpy as np

class deploy:
    def __init__(self, model_name):
        self.model_name = model_name
        self.provider = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(self.model_name, providers=self.provider)

    def image_preprocessing(self, input_shape, image):
        height, width = input_shape[2:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(width,height),interpolation = cv2.INTER_AREA)
        image = image / 255.0
        image = image.transpose(2,0,1)
        input_tensor = image[np.newaxis, :, :, :].astype(np.float32)
    
        return input_tensor,height,width


    def compute_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_intersection = max(0, min(x1_min, x2_min) - max(x1_max, x2_max))
        y_intersection = max(0, min(y1_min, y2_min) - max(y1_max, y2_max))
        intersection_area = x_intersection * y_intersection

        area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

        iou = intersection_area / (area_box1 + area_box2 - intersection_area)
        return iou

    def nms(self, boxes, scores, iou_threshold):
        selected_indices = []

        sorted_indices = np.argsort(scores)[::-1]

        while len(sorted_indices) > 0:
            best_index = sorted_indices[0]
            selected_indices.append(best_index)

            iou_values = [self.compute_iou(boxes[best_index], boxes[idx]) for idx in sorted_indices[1:]]
            filtered_indices = np.where(np.array(iou_values) < iou_threshold)[0]
            sorted_indices = sorted_indices[filtered_indices + 1]

        return selected_indices

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y                                                                                             

        
    def on_options(self):
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = True
        opt_session.enable_cpu_mem_arena = True
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        
    def run(self, video_path):
        model_inputs = self.ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        model_outputs = self.ort_session.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FRAME_WINDOW = st.image([])
        
        while True:
            
            ret, frame = video.read()

            if not ret or video.isOpened()==False:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor, h, w = self.image_preprocessing(model_inputs[0].shape, frame)

            output = self.ort_session.run(output_names, {input_names[0]: input_tensor})[0]

            predictions = np.squeeze(output).T
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > 0.6, :]
            scores = scores[scores > 0.6]

            boxes = predictions[:, :4]
                
            input_shape = np.array([w, h, w, h])
                
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([width, height, width, height])
            boxes = boxes.astype(np.int32)

            indices = self.nms(boxes, scores, 0.3)

            for (bbox, score) in zip(self.xywh2xyxy(boxes[indices]), scores[indices]):
                bbox = bbox.round().astype(np.int32).tolist()
                color = (0,255,0)
                cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            
            FRAME_WINDOW.image(frame)

        os.remove(video_path)
        video.release()


if __name__=="__main__":
    st.markdown("<h2 style='text-align: center; color: white;'>ONNX model deployment with GCP</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload video file here...", type=["mp4", "avi", "mov"])

    deployment = deploy("face_detection_model.onnx")
    deployment.on_options()
    
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            video_path = temp_file.name
            temp_file.write(file.read())
            deployment.run(video_path)
