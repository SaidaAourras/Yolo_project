from pandas.core import frame
from cv2 import VideoCapture
import cv2
from ultralytics import YOLO
import numpy as np
import json


model = YOLO('yolo11n.pt')

cap = VideoCapture("vid.mp4")

classes_v = ['car', 'bus', 'truck']

roi = np.array([
    (210,150),(500,150),(746,430),(8,430)
], np.int32)


vehicles_in_zone = {}
current_ids_in_zone = set()

frame_number = 0

events = []

fps = cap.get(cv2.CAP_PROP_FPS)

last_seen = []


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def detect_vehicles(frame):

    results = model.track(frame,persist=True,tracker="botsort_reid.yaml")

    return results


def process_vehicle(box,track_id,confidence,class_id,frame_number,timestamp):

    class_name = model.names[class_id]
    if class_name not in classes_v:
        return

    x1, y1, x2, y2 = box.xyxy[0]

    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)

    inside = cv2.pointPolygonTest(roi,(cx,cy),False)

    if inside >= 0:

        current_ids_in_zone.add(track_id)
        if track_id not in vehicles_in_zone:

            print(" Vehicle ID", track_id , "entered")
            vehicles_in_zone[track_id] = frame_number

            events.append({
                "event":"vehicle_entering_the_zone",
                "track_id":int(track_id),
                "zone": "Zone A",
                "timestamp":timestamp,
                "confidence": float(confidence)
            })


def process_exited_vehicles(frame_number, timestamp):

    exited_ids = set(vehicles_in_zone.keys()) - current_ids_in_zone
    for track_id in exited_ids:
        print("Vehicle ID", track_id , "exited")
        entry_frame = vehicles_in_zone[track_id]

        duration_seconds = round((frame_number - entry_frame)/fps,2 )

        events.append({
            "event":"vehicle_exiting_the_zone",
            "track_id":int(track_id),
            "zone": "Zone A",
            "timestamp":timestamp,
        })

        print("Vehicle ID",
            track_id,"stayed in zone for",
            duration_seconds,"seconds"
        )

        del vehicles_in_zone[track_id]


def process_results(results, frame_number, timestamp):
    result = results[0]

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, confidence, class_id in zip(result.boxes,track_ids,confidences,classes
        ):

            process_vehicle(box,track_id,confidence,class_id,frame_number,timestamp)


def save_events():

    with open("events.json" ,"w") as f:
        json.dump(events , f , indent=4)

def main():
    global cap
    global frame_number
    global events
    global current_ids_in_zone
    global vehicles_in_zone

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        "video_annotated.mp4",fourcc,fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:

        success, frame = cap.read()
        frame_number += 1
        timestamp = round(frame_number / fps, 2)


        cv2.polylines(frame,[roi],True,(0,255,0),2)

        results = detect_vehicles(frame)

        current_ids_in_zone.clear()

        process_results(results,frame_number,timestamp)

        process_exited_vehicles(frame_number,timestamp)

        annotat_frame = results[0].plot()

        # cv2.imshow("roi", roi)
        # cv2.imshow("frame", frame)

        cv2.imshow("annotate_frame",annotat_frame)

        # cv2.imshow('mask',mask)

        key = cv2.waitKey(30)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    save_events()


if __name__ == "__main__":
        main()


        