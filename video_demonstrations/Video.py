import os
import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from video_demonstrations.Frame import Frame
from pathlib import Path


class Video:
    def __init__(
        self,
        images_root,
        sr_prefix="SR",
        lr_prefix="LR",
        output_size=(800, 800),
        output_folder="data/visualisations/videos/",
        interpolation=Image.NEAREST,
        watermark_logo_path=None,
        watermark_logo_size=(75, 75),
        watermark_position=None,
    ):
        """ Initializes the Video class used to generate video demonstrations comparing 
        the super-resolved image and low-resolution images.

        Parameters
        ----------
        images_root : str
            The root path of the images.
        sr_prefix : str, optional
            The prefix of the super-resolved image file. The default is "SR".
        lr_prefix : str, optional
            The prefix of the low-resolution image files. The default is "LR".
        output_size : tuple, optional
            The size of the output video. The default is (800, 800).
        output_folder : str, optional
            The folder to save the video to. The default is "data/visualisations/videos/".
        interpolation : Image.InterpolationMode, optional
            The interpolation method to use when resizing the images. The default is Image.NEAREST.
        watermark_logo_path : str, optional
            The path to the watermark logo. The default is None.
        watermark_logo_size : tuple, optional
            The size of the watermark logo. The default is (75, 75).
        watermark_position : tuple, optional
            The position of the watermark logo. The default is None.
        """
        self.images_root = images_root
        self.sr_prefix = sr_prefix
        self.lr_prefix = lr_prefix
        self.output_size = output_size
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        self.interpolation = interpolation
        self.frames = []
        self.load_images()
        self.set_image_size()

        self.watermark_logo_path = watermark_logo_path
        self.watermark_logo_size = watermark_logo_size
        self.watermark_position = watermark_position
        self.load_watermark()

    def set_image_size(self):
        """ Sets the image size of the video, based on the size of the first image in the image group.
        """
        self.frame_width, self.frame_height = self.image_group["Low Resolution 1"].size

    def load_watermark(self):
        """ Loads the watermark/logo from the path specified in the constructor, and resizes it to the size specified in the constructor, if any.
        """
        if self.watermark_logo_path is not None:
            self.watermark_logo = Image.open(self.watermark_logo_path).resize(
                self.watermark_logo_size, Image.BICUBIC
            )
            if self.watermark_position is None:
                self.watermark_position = (  # Bottom right corner, 25px from the right and 25px from the bottom
                    self.frame_width - self.watermark_logo.size[0] - 25,
                    self.frame_height - self.watermark_logo.size[1] - 25,
                )
        else:
            self.watermark_logo = None

    def load_images(self):
        """ Loads the images from the root path specified in the constructor, and resizes them to the size specified in the constructor.
        Uses the check_and_load_images_root method to check if the root contains the correct images with the correct prefixes, and if not, raises an exception.
        """
        image_group = {}
        self.check_and_load_images_root()
        image_group["Super-Resolved"] = Image.open(self.sr_image).resize(
            self.output_size, self.interpolation
        )
        image_group.update(
            {
                f"Low Resolution {i+1}": Image.open(self.lr_images[i]).resize(
                    self.output_size, self.interpolation
                )
                for i in range(len(self.lr_images))
            }
        )
        self.image_group = image_group

    def check_and_load_images_root(self):
        """ Checks if the root contains the correct images with the correct prefixes and loads them, if not, raises an exception.

        Raises
        ------
        Exception
            If the root does not contain the correct images with the correct prefixes.
        Exception
            If the root does not contain the correct images with the correct prefixes.
        """
        sr_image, lr_images = None, []
        valid_image_types = (".png", ".jpg", ".jpeg")
        files_in_root = os.listdir(self.images_root)
        if len(files_in_root) == 0:
            raise Exception("No images in root")

        for file in files_in_root:
            if file.startswith(self.sr_prefix) and file.endswith(valid_image_types):
                sr_image = os.path.join(self.images_root, file)
            elif file.startswith(self.lr_prefix) and file.endswith(valid_image_types):
                lr_images.append(os.path.join(self.images_root, file))
        if sr_image == None or len(lr_images) == 0:
            raise Exception(
                "Root doesn't contain a valid SR image or LR images. Check the prefixes, and make sure the images are named correctly."
            )
        self.sr_image = sr_image
        self.lr_images = lr_images

    def watermark_frames(self):
        """ Watermarks the video frames if a watermark logo is specified in the constructor.
        """
        if self.watermark_logo is None or self.watermark_logo_size is None:
            return
        for frame in self.frames:
            frame.paste(
                self.watermark_logo, self.watermark_position, self.watermark_logo
            )

    def save_frames_to_video(self, video_path="video.mp4", fps=60, overwrite=False):
        """ Saves the current video frames to a video file.

        Parameters
        ----------
        video_path : str, optional
            The filename under which to save the video to in the output folder, by default "video.mp4".
        fps : int, optional
            The frames per second of the video. The default is 60.
        overwrite : bool, optional
            Whether to overwrite the video if it already exists. The default is False.
        """
        self.watermark_frames()
        video_path = str(Path(self.output_folder, video_path))
        if overwrite and os.path.isfile(video_path):
            os.remove(video_path)
        if isinstance(self.frames[0], Image.Image):
            self.frames = self.convert_frames_to_opencv()

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        video = cv2.VideoWriter(
            video_path, fourcc, fps, (self.frame_width, self.frame_height)
        )
        for frame in tqdm(self.frames, desc=f"Writing frames to {video_path}"):
            video.write(frame)

        video.release()

    def convert_frames_to_opencv(self):
        """ Converts the PIL.Image frames to OpenCV frames.

        Returns
        -------
        list
            The list of OpenCV frames.
        """
        return [
            cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            for frame in tqdm(self.frames, desc="Converting frames to OpenCV")
        ]

    def save_frames_from_files_to_video(self, path, video_path="test.mp4", fps=60):
        """ Saves the frames from the files in the path specified to a video file.
        Used when there are too many frames to load into memory/the video is too long.

        Parameters
        ----------
        path : str
            The path to the folder containing the frames.
        video_path : str, optional
            The filename under which to save the video to in the output folder, by default "test.mp4".
        fps : int, optional
            The frames per second of the video. The default is 60.
        """
        frames = [Image.open(os.path.join(path, file)) for file in os.listdir(path)]
        self.watermark_frames()
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        video = cv2.VideoWriter(
            video_path, fourcc, fps, (self.frame_width, self.frame_height)
        )
        for frame in frames:
            video.write(frame)

        video.release()

    def generate_zoom_into_group(self, zoom_x, zoom_y, box_size=50):
        """ Generates a zoom into for a group of images.

        Parameters
        ----------
        zoom_x : int
            The x-coordinate of the zoom target.
        zoom_y : int
            The y-coordinate of the zoom target.
        box_size : int, optional
            The size of the box to zoom into. The default is 50.
        """
        self.zoom_x = zoom_x
        self.zoom_y = zoom_y
        self.box_size = box_size

        box = self.generate_zoom_box(zoom_x, zoom_y, box_size)
        self.generate_zoom_into_group_for_box(box)

    def generate_zoom_box(self, zoom_x, zoom_y, box_size=50):
        """ Generates a zoom box for given x and y target coordinates.

        Parameters
        ----------
        zoom_x : int
            The x-coordinate of the zoom target.
        zoom_y : int
            The y-coordinate of the zoom target.
        box_size : int, optional
            The size of the box to zoom into. The default is 50.

        Returns
        -------
        list
            The list of coordinates of the zoom box.
        """
        self.zoom_box = (
            zoom_x - box_size,
            zoom_y - box_size,
            zoom_x + box_size,
            zoom_y + box_size,
        )
        return self.zoom_box

    def generate_zoom_into_group_for_box(self, zoom_box):
        """ Generates a zoom into animation for the given zoom box.

        Parameters
        ----------
        zoom_box : list
            The list of coordinates of the zoom box.
        """
        self.zoomed_into_group = {
            name: frame.crop(zoom_box).resize(self.output_size, self.interpolation)
            for name, frame in self.image_group.items()
        }
        self.full_size_image_group = {
            name: image.copy() for name, image in self.image_group.items()
        }

    def zoom_into(self):
        """ Switches to the zoomed-in image group.
        """
        self.image_group = self.zoomed_into_group

    def zoom_out(self):
        """ Switches to the full-size/zoomed-out image group.
        """
        self.image_group = self.full_size_image_group
