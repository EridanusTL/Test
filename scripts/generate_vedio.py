import cv2
import os


def images_to_video(image_folder, video_name, fps):
    file_list = os.listdir(image_folder)
    images = sorted(
        file_list, key=lambda x: os.path.getmtime(os.path.join(image_folder, x))
    )

    images = [
        "frame_" + str(idx) + ".jpg" for idx in range(len(os.listdir(image_folder)))
    ]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    folder_path = "dataset/video/predict"
    video_name = "output/output.mp4"
    frames_per_second = 6  # 调整为你需要的帧率

    images_to_video(folder_path, video_name, frames_per_second)
