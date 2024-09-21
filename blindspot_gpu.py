import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import os
import pyttsx3
import threading
import multiprocessing

# global settings for audio output
tt_to_speech = pyttsx3.init()
tt_to_speech.setProperty('rate', 150)
tt_to_speech.setProperty('volume', 0.9)

# Cords corresponding to i/p vdo 
polygon = np.array([
[286, 716],[294, 16],[14, 16],[14, 704]
])

# gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(torch.cuda.is_available())

video_path = "test_vdo_city.m4v"

def voice_output(text):
    tt_to_speech.say(text)
    tt_to_speech.runAndWait()

def main():
    video_info = sv.VideoInfo.from_video_path(video_path)  
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )
    
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=6, text_thickness=6, text_scale=4)

    model = YOLO("yolov8n.pt")
    for result in model.track(source=video_path, stream=True, agnostic_nms=True, device=0):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)


        # prevent the error when no object is detected in the frame
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        #filtering the detections with higher confidence 
        detections = detections[(detections.confidence >= 0.45)]
        
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

        zone.trigger(detections=detections)

        # audio alerts for blind spot
        print(f'objects in blind spot: {zone.current_count} ')
        if(zone.current_count != 0 and zone.current_count > 1): 
            thread_aud = threading.Thread(target=voice_output, args=(f'{zone.current_count} objects at the blind spot',))
            thread_aud.start()
        if(zone.current_count == 1):
            thread_aud = threading.Thread(target=voice_output, args=(f'{zone.current_count} object at the blind spot',))
            thread_aud.start()

        
        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.2)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=4, text_thickness=4, text_scale=2)
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("BLIND SPOT ASSIST", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    proc_main = multiprocessing.Process(target=main)
    proc_main.start()
    proc_main.join()

    