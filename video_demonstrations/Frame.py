from PIL import Image, ImageDraw, ImageFont


class Frame:
    """ A Frame class that allows single frame manipulation for the demonstration videos. """

    def __init__(
        self, frame, font_path="video_demonstrations/font/BAHNSCHRIFT.TTF", font_size=64
    ):
        """ Initialise the Frame class.

        Parameters
        ----------
        frame : PIL.Image
            The frame to be manipulated.
        font_path : str, optional
            The path to the font file. The default is "video_demonstrations/font/BAHNSCHRIFT.TTF".
        font_size : int, optional
            The font size. The default is 64.

        Raises
        ------
        Exception
            If the frame is not a PIL.Image.
        """
        if isinstance(frame, Image.Image):
            self.original_frame = frame.copy()
            self.frame = frame.copy()
        else:
            raise Exception("Frame must be a PIL Image")
        self.font_path = font_path
        self.font_size = font_size
        self.load_font()
        self.width, self.height = self.frame.size

    def revert_to_original_frame(self):
        """ Revert the frame to the original frame. """
        self.frame = self.original_image.copy()

    def load_font(self):
        """ Load the font. """
        self.font = ImageFont.truetype(self.font_path, self.font_size)

    def change_font_size(self, font_size):
        """ Change the font size.

        Parameters
        ----------
        font_size : int
            The new font size.
        """
        self.font_size = font_size
        self.load_font()

    def create_mask(self, width, height, x=0, y=0):
        """ Create a mask for the frame.

        Parameters
        ----------
        width : int
            The width of the mask.
        height : int
            The height of the mask.
        x : int, optional
            The starting (upper-left) x-coordinate of the mask. The default is 0.
        y : int, optional
            The starting (upper-left) y-coordinate of the mask. The default is 0.

        Returns
        -------
        PIL.Image
            The mask.
        """
        mask_im = Image.new("L", self.frame.size, 0)
        draw = ImageDraw.Draw(mask_im)
        draw.rectangle((x, y, width + x, height + y), fill=255)
        return mask_im

    def generate_title_frames(self, title, fps, duration):
        """ Generates a list of static frames with a title.

        Parameters
        ----------
        title : str
            The title to be written on the frames.
        fps : int
            The frames per second.
        duration : int
            The duration of the title in seconds.

        Returns
        -------
        list
            A list of frames.
        """
        number_of_frames = duration * fps
        title_frame = self.write_on_frame_center(title)
        return [title_frame for frame in range(number_of_frames)]

    def write_on_frame(self, text, position, color=(255, 255, 255)):
        """ Write text on the frame.

        Parameters
        ----------
        text : str
            The text to be written.
        position : tuple
            The position of the text.
        color : tuple, optional
            The color (RGB) of the text. The default is (255, 255, 255).

        Returns
        -------
        PIL.Image
            The frame with the text written on it.
        """
        draw = ImageDraw.Draw(self.frame)
        draw.text(position, text, color, font=self.font)
        return self.frame

    def write_on_frame_center(self, text):
        """ Write text in the center of the frame.

        Parameters
        ----------
        text : str
            The text to be written.

        Returns
        -------
        PIL.Image
            The frame with the text written on it.
        """
        draw = ImageDraw.Draw(self.frame)
        text_width, _ = draw.textsize(text, font=self.font)
        image_width, image_height = self.frame.size
        text_position = ((image_width - text_width) // 2, int(image_height * 0.9))
        self.write_on_frame(text, text_position)
        return self.frame

    def write_on_frame_top(self, text, font_size=48):
        """ Write text on the top of the frame.

        Parameters
        ----------
        text : str
            The text to be written.
        font_size : int, optional
            The font size. The default is 48.

        Returns
        -------
        PIL.Image
            The frame with the text written on it.
        """
        self.change_font_size(font_size)
        draw = ImageDraw.Draw(self.frame)
        text_width, _ = draw.textsize(text, font=self.font)
        image_width, image_height = self.frame.size
        text_position = ((image_width - text_width) // 2, int(image_height * 0.05))
        self.write_on_frame(text, text_position)
        return self.frame
