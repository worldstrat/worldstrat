from PIL import Image
from video_demonstrations.Frame import Frame
import numpy as np
import os


class Scenario:
    @staticmethod
    def generate_default_scenarios(video):

        """ Generates the default scenarios for the video.

        The default scenarios are:
        - A single revisit comparison video (between the first revisit and the super resolved image).
        - A comparison between vertically merged low resolution images and the super resolved image.
        - A comparison between horizontally merged low resolution images and the super resolved image.

        All of the scenarios have built in transitions showcasing all of the low-resolution images.
        If an annotation mask is provided, the images are also annotated with the mask.

        The three scenarios are saved to three files in the output_path:
        - single_revisit_comparison.mp4
        - merged_vertical_comparison.mp4
        - merged_horizontal_comparison.mp4

        And finally, the horizontally merged comparison scenario is returned as a list of frames.

        Returns
        -------
        list
            A list of frames, each frame being a single frame of the horizontally merged comparison scenario.
        """
        reference_lowres_image = video.image_group["Low Resolution 1"]
        super_res_image = video.image_group["Super-Resolved"]

        if "Annotation Mask" in video.image_group.keys():
            lowres_image_names = [
                name
                for name in video.image_group.keys()
                if name.startswith("Low Resolution")
            ]
            for lowres_image in lowres_image_names:
                annotation_mask = video.image_group["Annotation Mask"]
                video.image_group[lowres_image].paste(
                    annotation_mask, (0, 0), annotation_mask
                )
            super_res_image.paste(annotation_mask, (0, 0), annotation_mask)
        lowres_images = [
            image
            for name, image in video.image_group.items()
            if name.startswith("Low Resolution")
        ]

        horizontally_merged_revisits_preview = Frame(
            Scenario.generate_merged_preview_horizontal(video), font_size=32
        )

        horizontally_merged_revisits_preview.write_on_frame(
            f"All {len(lowres_images)} Revisits", (580, 30)
        )
        lowres_flash_frames = Scenario.generate_lowres_preview_reel(video)
        single_revisit_comparison = Scenario.single_revisit_comparison(
            reference_lowres_image, super_res_image, repetition=2
        )
        transition_to_stripes = Scenario.generate_transition_to_stripes(
            lowres_images, horizontally_merged_revisits_preview.frame
        )
        merged_vertical_revisits_comparison = Scenario.merged_vertical_revisits_comparison(
            horizontally_merged_revisits_preview.frame, super_res_image, repetition=2
        )
        merged_horizontal_revisits_comparison = Scenario.merged_horizontal_revisits_comparison(
            horizontally_merged_revisits_preview.frame, super_res_image, repetition=2
        )

        video.frames = lowres_flash_frames + single_revisit_comparison
        video.save_frames_to_video("single_revisit_comparison.mp4")

        video.frames = (
            lowres_flash_frames
            + transition_to_stripes
            + merged_vertical_revisits_comparison
        )
        video.save_frames_to_video("merged_vertical_comparison.mp4")

        video.frames = (
            lowres_flash_frames
            + transition_to_stripes
            + merged_horizontal_revisits_comparison
        )
        video.save_frames_to_video("merged_horizontal_comparison.mp4")

        return (
            lowres_flash_frames
            + transition_to_stripes
            + merged_horizontal_revisits_comparison
            + transition_to_stripes[::-1]
        )

    @staticmethod
    def single_revisit_comparison(reference_image, super_res_image, repetition=4):
        """ Generates a single revisit comparison between the reference image and the super-res image.

        Parameters
        ----------
        reference_image : PIL.Image
            The reference image.
        super_res_image : PIL.Image
            The super-res image.
        repetition : int, optional
            The number of times the comparison is repeated. The default is 4.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        from_frame = Frame(reference_image, font_size=32)
        to_frame = Frame(super_res_image, font_size=32)
        from_frame.write_on_frame("Low Resolution (Sentinel-2)", (390, 30))
        to_frame.write_on_frame("Merged Superresolved", (40, 30))
        single_revisit_comparison = Scenario.slide_from_to(
            from_frame, to_frame, slider_length_per_frame=5, name=None, rollback=True
        )
        single_revisit_comparison *= repetition

        return single_revisit_comparison

    @staticmethod
    def generate_transition_to_stripes(lowres_images, striped_image):
        """ Generates a transition from the lowres images to the striped revisits image.

        Parameters
        ----------
        lowres_images : list
            A list of PIL.Image objects.
        striped_image : PIL.Image
            The striped image of all revisits.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        from_frame = Frame(lowres_images[0], font_size=32)
        to_frame = Frame(striped_image, font_size=32)
        transition_to_stripes = Scenario.slide_from_to(
            from_frame,
            to_frame,
            slider_length_per_frame=5,
            name=None,
            rollback=False,
            start_offset=0,
        )
        return transition_to_stripes

    @staticmethod
    def merged_vertical_revisits_comparison(
        striped_image, super_res_image, repetition=4
    ):
        """ Generates a vertically sliding comparison between the striped revisits image and the super-res image.

        Parameters
        ----------
        striped_image : PIL.Image
            The striped image of all revisits.
        super_res_image : PIL.Image
            The super-res image.
        repetition : int, optional
            The number of times the comparison is repeated. The default is 4.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """

        from_frame = Frame(striped_image, font_size=32)
        to_frame = Frame(super_res_image, font_size=32)

        to_frame.write_on_frame("Merged Superresolved", (40, 30))

        rotated_from_frame = Frame(from_frame.frame.copy().rotate(90))
        rotated_to_frame = Frame(to_frame.frame.copy().rotate(90))

        merged_vertical_revisits_comparison = Scenario.slide_from_to(
            rotated_from_frame,
            rotated_to_frame,
            slider_length_per_frame=5,
            name=None,
            rollback=True,
            start_offset=0,
        )
        merged_vertical_revisits_comparison = [
            x.rotate(-90) for x in merged_vertical_revisits_comparison
        ]
        merged_vertical_revisits_comparison *= repetition

        return merged_vertical_revisits_comparison

    @staticmethod
    def merged_horizontal_revisits_comparison(
        striped_image, super_res_image, repetition=3
    ):
        """ Generates a horizontally sliding comparison between the striped revisits image and the super-res image.

        Parameters
        ----------
        striped_image : PIL.Image
            The striped image of all revisits.
        super_res_image : PIL.Image
            The super-res image.
        repetition : int, optional
            The number of times the comparison is repeated. The default is 3.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        from_frame = Frame(striped_image, font_size=32)
        to_frame = Frame(super_res_image, font_size=32)

        to_frame.write_on_frame("Merged Superresolved", (40, 30))
        merged_horizontal_revisits_comparison = Scenario.slide_from_to(
            from_frame,
            to_frame,
            slider_length_per_frame=5,
            name=None,
            rollback=True,
            start_offset=0,
        )
        merged_horizontal_revisits_comparison *= repetition
        return merged_horizontal_revisits_comparison

    @staticmethod
    def slide_from_to(
        from_frame,
        to_frame,
        slider_length_per_frame=5,
        name=None,
        rollback=True,
        start_offset=0,
        duration=None,
    ):
        """ Slides from one frame to another.

        Parameters
        ----------
        from_frame : Frame
            The frame to slide from.
        to_frame : Frame
            The frame to slide to.
        slider_length_per_frame : int, optional
            The length of the slider in pixels. The default is 5.
        name : str, optional
            The name of the scenario. The default is None.
        rollback : bool, optional
            Whether the slider should be rolled back. The default is True.
        start_offset : int, optional
            The start offset of the slider in pixels. The default is 0.
        duration : int, optional
            The max total duration of the scenario in seconds. The default is None.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """

        font_size = int(64 * (from_frame.width / 800))
        from_frame.change_font_size(font_size)
        to_frame.change_font_size(font_size)

        if duration is not None:
            slider_length_per_frame = from_frame.width // duration
        if name is None:
            title_frames = []
        else:
            title_frames = from_frame.generate_title_frames(name, 60, 2)
        slider_frames = [
            Scenario.slider_from_to_step(
                from_frame, to_frame, slider_step, start_offset
            )
            for slider_step in range(0, to_frame.width + 1, slider_length_per_frame)
        ]
        slider_frames += [slider_frames[-1].copy() for _ in range(30)]
        if rollback:
            slider_frames += slider_frames[::-1]
            slider_frames += [from_frame.frame.copy() for _ in range(60)]
        return title_frames + slider_frames

    @staticmethod
    def slider_from_to_step(
        from_frame, to_frame, slider_step, start_position_x=0, start_position_y=0
    ):
        """ Generates a single frame/step for the slider from one frame to another.

        Parameters
        ----------
        from_frame : Frame
            The frame to slide from.
        to_frame : Frame
            The frame to slide to.
        slider_step : int
            The step of the slider.
        start_position_x : int, optional
            The start position of the slider in the x-axis. The default is 0.
        start_position_y : int, optional
            The start position of the slider in the y-axis. The default is 0.

        Returns
        -------
        PIL.Image
            The generated frame.
        """
        slider_mask = from_frame.create_mask(
            slider_step, to_frame.height, start_position_x, start_position_y
        )
        frame = from_frame.frame.copy()
        frame.paste(to_frame.frame, (start_position_x, start_position_y), slider_mask)
        return frame

    @staticmethod
    def generate_default_scenario(video, video_name="video.mp4"):
        """ Generates the default video scenario:
        - Show all of the revisits for 1 second each.
        - Slide from the first revisit to the super-res image.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.
        video_name : str, optional
            The output name of the video. The default is "video.mp4".

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        video.frames = []
        previous_image = video.image_group["Low Resolution 1"]
        for name, image in video.image_group.items():
            if name.startswith("Low Resolution"):
                if previous_image is not None:
                    video.frames += Scenario.generate_transition(
                        previous_image, image, 1
                    )
                frame_from = Frame(image)
                frame_to = Frame(video.image_group["Super-Resolved"])
                video.frames += Scenario.slide_from_to(frame_from, frame_to, 5, name)
                previous_image = image
        return video.frames

    @staticmethod
    def generate_transition(image_from, image_to, duration, fps=60):
        """ Generates a transition between two images.

        Parameters
        ----------
        image_from : PIL.Image
            The image to slide from.
        image_to : PIL.Image
            The image to slide to.
        duration : int
            The duration of the transition in seconds.
        fps : int, optional
            The frames per second of the video. The default is 60.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        number_of_frames = duration * fps
        return [
            Image.blend(image_from, image_to, alpha)
            for alpha in np.arange(0, 1, 1 / number_of_frames)
        ]

    @staticmethod
    def generate_lowres_preview_reel(video, duration_per_image=0.5, fps=60):
        """ Generates a low-res preview reel that shows each low-res frame for a certain duration.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.
        duration_per_image : int, optional
            The duration of each image in seconds. The default is 0.5.
        fps : int, optional
            The frames per second of the video. The default is 60.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        reel_frames = []
        for name, image in video.image_group.items():
            if name.startswith("Low Resolution"):
                titled_image = Frame(image)
                titled_image = titled_image.write_on_frame_top(name)
                reel_frames += [titled_image] * int(duration_per_image * fps)
        reel_frames += [reel_frames[0] for _ in range(int(duration_per_image * fps))]
        return reel_frames

    @staticmethod
    def generate_white_flash(video, duration, fps=60):
        """ Generates a white flash (transition) for a certain duration.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.
        duration : int
            The duration of the transition in seconds.
        fps : int, optional
            The frames per second of the video. The default is 60.

        Returns
        -------
        _type_
            _description_
        """
        size = video.image_group["Low Resolution 1"].size
        number_of_frames = duration * fps
        return [
            Image.new(
                "RGB",
                size,
                (
                    (255 // number_of_frames) * frame,
                    (255 // number_of_frames) * frame,
                    (255 // number_of_frames) * frame,
                ),
            )
            for frame in range(1, number_of_frames)
        ]

    @staticmethod
    def showcase_all_images_slider(video):
        """ Generates a slider that shows all images.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        frames = []
        start_image = Frame(video.image_group["Super-Resolved"])
        start_image_with_title = Frame(video.image_group["Super-Resolved"].copy())
        start_image_with_title.write_on_frame_center("Super-Resolved")
        for name, image in video.image_group.items():
            if name.startswith("Low Resolution"):
                low_resolution_image_with_title = Frame(image.copy())
                low_resolution_image_with_title.write_on_frame_center(name)
                frames += video.slide_from_to(
                    start_image, low_resolution_image_with_title, 5, rollback=False
                )
                start_image = low_resolution_image_with_title
        return frames

    @staticmethod
    def generate_merged_preview_vertical(video):
        """ Generates a merged preview reel that shows the vertically merged low-res images.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        merged_image = Image.new("RGB", (video.frame_width, video.frame_height), 0)
        width_per_image = video.frame_width / (len(video.image_group) - 1)

        image_index = 0
        for name, image in video.image_group.items():
            if name.startswith("Low Resolution"):
                lowres_image = image.copy()
                lowres_image = lowres_image.crop(
                    (
                        int(image_index * width_per_image),
                        0,
                        int((image_index + 1) * width_per_image),
                        video.frame_height,
                    )
                )
                merged_image.paste(
                    lowres_image, (int(image_index * width_per_image), 0)
                )
                image_index += 1
        return merged_image

    @staticmethod
    def generate_merged_preview_horizontal(video):
        """ Generates a merged preview reel that shows the horizontally merged low-res images.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        merged_image = Image.new("RGB", (video.frame_width, video.frame_height), 0)
        height_per_image = video.frame_height / (len(video.image_group) - 1)
        image_index = 0
        for name, image in video.image_group.items():
            if name.startswith("Low Resolution"):
                lowres_image = image.copy()
                lowres_image = lowres_image.crop(
                    (
                        0,
                        int(image_index * height_per_image),
                        video.frame_width,
                        int((image_index + 1) * height_per_image),
                    )
                )
                merged_image.paste(
                    lowres_image, (0, int(image_index * height_per_image))
                )
                image_index += 1
        return merged_image

    @staticmethod
    def generate_zoom_into_animation(
        video, starting_image="Low Resolution 1", fps=60, zoom_duration=3
    ):
        """ Generates a zoom-in animation to the zoom box coordinates defined in the Video object.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for and the zoom box coordinates.
        starting_image : str, optional
            The name of the image to start the animation with. The default is "Low Resolution 1".
        fps : int, optional
            The frames per second of the video. The default is 60.
        zoom_duration : int, optional
            The duration of the zoom in animation in seconds. The default is 3.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        start = (0, 0) + video.output_size
        zoom_coordinates = video.zoom_box
        starting_image = video.image_group[starting_image].copy()
        steps = fps * zoom_duration
        zoom_steps = [(x - y) / steps for x, y in zip(start, zoom_coordinates)]
        start, zoom_steps = np.array(start), np.array(zoom_steps)

        zoom_crops = np.array(
            [np.array(start - i * zoom_steps) for i in range(steps + 1)]
        )

        zoom_frames = [
            starting_image.copy()
            .crop(zoom_crops[i])
            .resize(starting_image.size, Image.NEAREST)
            for i in range(steps + 1)
        ]
        return zoom_frames

    @staticmethod
    def generate_zoom_out_animation(
        video, starting_image="Low Resolution 1", fps=60, zoom_duration=3
    ):
        """ Generates a zoom-out animation out of the zoom box coordinates defined in the Video object.

        Parameters
        ----------
        video : Video
            The video object (with the loaded images) to generate the scenario for and the zoom box coordinates.
        starting_image : str, optional
            The name of the image to start the animation with. The default is "Low Resolution 1".
        fps : int, optional
            The frames per second of the video. The default is 60.
        zoom_duration : int, optional
            The duration of the zoom out animation in seconds. The default is 3.

        Returns
        -------
        list
            A list of PIL.Image objects.
        """
        return Scenario.generate_zoom_into_animation(
            video, starting_image, fps, zoom_duration
        )[::-1]

