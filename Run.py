import albumentations as A
from pathlib import Path
from ultralytics import YOLO
from SRD import srd_pipeline
from Augmentor_config import transform


# =====================================================================
# Main Function
# =====================================================================
if __name__ == "__main__":
	# Integrating SRD and DDA
    combined_transforms = list(srd_pipeline.transforms) + list(transform.transforms)
    final_aug_pipeline = A.Compose(
        transforms=combined_transforms,
        keypoint_params=transform.processors["keypoints"].params if "keypoints" in transform.processors else None,
        strict=True,
    )
	
	# ==========================================
    #  Select the dataset, including:
    # "CrowdPose.yaml", "coco2017.yaml", "mpii-pose.yaml", "hand-keypoints.yaml"
    # ==========================================
    dataset = Path("~/run/datasets/CrowdPose/mpii-pose.yaml").expanduser()

    # ==========================================
    #  Teacher Model Training 
    # ==========================================
    teacher = YOLO("yolo26s-pose.pt")  
    name = "yolo26s-SRDv3-Aug-mpii"

    train_params = {
            "data": dataset,
            "epochs": 100,
            "imgsz": 640,   
            "batch": 64,
            "augmentations": final_aug_pipeline,
            "patience": 30,
    }
    teacher.train(**train_params)

    # ===============================================
    # Student Model Training 
    # ===============================================
    student = YOLO("yolo26n-pose.pt")
    teacher = Path("yolo26s-SRDv3-Aug-mpii/weights/best.pt")
    name = "yolo26n-SRD-Aug-Distill-mpii"
    
    student_params = {
        "data": dataset,
        "epochs": 100,
        "imgsz": 640, 
        "batch": 64,
        "augmentations": final_aug_pipeline,
        "patience": 30,
        "distill_model": teacher
    }
    student.train(**student_params)





