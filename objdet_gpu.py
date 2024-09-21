import cv2
from ultralytics import YOLO
import supervision as sv
import torch
import os
import pyttsx3
import threading
import multiprocessing

# global settings for audio output
tt_to_speech = pyttsx3.init()
tt_to_speech.setProperty('rate', 150)
tt_to_speech.setProperty('volume', 0.9)

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(torch.cuda.is_available())

video_path = "test_vdo_city.m4v"
# video_path = '0'

def voice_output(text):
    tt_to_speech.say(text)
    tt_to_speech.runAndWait()

def main():
    video_info = sv.VideoInfo.from_video_path(video_path)  
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )
    model = YOLO("yolov8l.pt")

    for result in model.track(source=0, stream=True, agnostic_nms=True, device=0):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)


        # prevent the error when no object is detected in the frame
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        #filtering the detections with higher confidence 
        detections = detections[(detections.confidence >= 0.45)]
        
        # audio alerts for objects
        for detection in detections:
            if detection[2] == 9: 
               print("traffic light ahead")
               thread_signal_audio = threading.Thread(target=voice_output, args=('Watch out for traffic signal',))
               thread_signal_audio.start()
                
            if detection[2] == 11:
               print("stop sign ahead")
               thread_signal_audio = threading.Thread(target=voice_output, args=('Watch out for stop signal',))
               thread_signal_audio.start()


        labels = [
            f"{tracker_id} {model.model.names[class_id]} {str(round((float(confidence) * 100),  1))}%"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.2)
        
        frame = box_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow("OBJECT MONITOR ", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    proc_main = multiprocessing.Process(target=main)
    proc_main.start()
    proc_main.join()
