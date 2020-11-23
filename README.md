# Traffic Counter

![car counting](./screenshots/177.0_screenshot.jpeg)
![car counting](./screenshots/177.0_thresh.jpeg)

This is the result of my undergraduate summer of research (2015) for the computer science department at Berea College. The OpenCV library is used to detect moving objects in a video feed by background subtraction and edge detection. 

No machine learning or fancy AI is being done here. This was mainly to keep the processing requirements low.

The system counts counts the number of cars passing the road. The project was developed on the original Raspberry Pi and therefore it needed to be fast to run. However, a faster device is recommended. 

The project was recently updated to use Python 3.8 and OpenCV 4.4.0.

You can check the paper report or the blog posts I made at the time to get a better idea about the motivation for the project.

## How to Use
There are several command line arguments that can given to the script. The information here might be outdated, so the safest way to check the available commands is by running:

```sh
python main.py --help
```

Which will return something like:

```
Finds the contours on a video file

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  A video filename or path. Works better with .avi
                        files. If no path or name is provided, the camera will
                        be used instead.
  -a MINAREA, --minArea MINAREA
                        The minimum area (in pixels) to draw a bounding box
  -d DIRECTION DIRECTION, --direction DIRECTION DIRECTION
                        A character: H or V representing the orientation of
                        the count line. H is horizontal, V is vertical. If not
                        provided, the default is horizontal. The second
                        parameter is a float number from 0 to 1 indicating the
                        place at which the line should be drawn.
  -n NUMCOUNT, --numCount NUMCOUNT
                        The number of contours to be detected by the program.
  -w WEBCAM [WEBCAM ...], --webcam WEBCAM [WEBCAM ...]
                        Allows the user to specify which to use as the video
                        source
  --rgb                 Boolean flag to use rbg colors.
  -vo VIDEO_OUT, --video_out VIDEO_OUT
                        Provide a video filename to output
  -vw VIDEO_WIDTH, --video_width VIDEO_WIDTH
                        Videos will be resized to this width. Height will be
                        computed automatically to preserve aspect ratio
  -vp VIDEO_PARAMS VIDEO_PARAMS, --video_params VIDEO_PARAMS VIDEO_PARAMS
                        Provide video codec and extension (in that order) for
                        the output video. Example: `--video_params mjpg avi`.
                        Default values are mjpg and avi
  -sf STARTING_FRAME, --starting_frame STARTING_FRAME
                        Select the starting frame for video analysis (default
                        is 10). All frames before that will still be used for
                        the background average

```

The most important argument is `-p` which gives the script the name of the file to analyze (use live camera video is not working yet). The flag `-d` is used to indicate the direction and position of the counting line. Like:

```sh
python main.py -p <path_to_your_video> -d v 0.5
```

The desired video width can also be provided with the flag `-vw`. This will also affect the size of the output video. 

```sh
python main.py -p <path_to_your_video> -vw 640
```

To get videos saved, use the flag `-vo` and some text that will be included in the output file's filename. For example:

```sh
python main.py -p <path_to_your_video> -vo my_video_without_extensions
```

Keep in mind that the default values for the file extension and video codec might not work for your system. Use tag `-vp` to change the default codec and extension combination. The codec is a 4-letter code (i.e. `MJPG`, `XDIV`, `DIVX`,etc). The extension can be something like `avi`, `mp4`, etc. For example:

```sh
python main.py -p <path_to_your_video> -vo some_video_name -vp MJPG avi
```

## Interface
I have updated the project and moved away from the original script. In the new one, the computer vision parts are handled in a class TrafficCoutner in traffic_counter.py. To run the script, you must run main.py with a combination of parameters. For example:

```sh
python main.py -p <path_to_your_video> -d V 0.5 
```

The `-p` parameter indicates a path to the video to be analyzed. `-d` is to indicate direction and position of the counting line. A `V` parameter is for a vertical line, expecting that cars are moving horizontally. The float 0.5 after `V` is the position of the line in the screen.

There are other parameters that can be modified, but as of now, I have not included a way to change them once the script starts. 

Once the script runs, a frame from the video will be displayed and you can click on several points on the frame to select an area of interest to calculate. Everything outside the selected area will be ignored. To proceed to the next part, press `q` or `enter` on the keyboard.

![Initial cropping](./screenshots/roi_mask_1207.0.jpeg)
![after applying mask](./screenshots/screenshot_1207.0.jpeg)

## Blog Post

As part of the research requirements, I wrote blog posts describing the progress made each week. It can be found
at:

[blog series](https://andrescscresearch.wordpress.com/)

## Paper Draft

A paper report of the work is also included in the repository:


[PaperReport_traffic_counter.pdf](./PaperReport_traffic_counter.pdf)

## Other links
I will be posting Python videos to my YouTube channel, among other projects. If you are interested in checking it out, here is the link:

[Andres Berejnoi Channel](https://www.youtube.com/andresberejnoi)
