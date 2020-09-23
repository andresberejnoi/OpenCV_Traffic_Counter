# Traffic Counter

This is the result of my undergraduate summer of research (2015) for the computer science department at Berea College. The OpenCV library is used to detect moving objects in a video feed by background subtraction and edge detection. 

No machine learning or fancy AI is being done here. This was mainly to keep the processing requirements low.

The system counts counts the number of cars passing the road. The project was developed on the original Raspberry Pi and therefore it needed to be fast to run. However, a faster device is recommended. 

The project was recently updated to use Python 3.8 and OpenCV 4.4.0.

You can check the paper report or the blog posts I made at the time to get a better idea about the motivation for the project.

## Interface

The script is very simple. At the beginning, you run the script and indicate a filename of a video to operate. If there is a default camera in the system, for example a web cam on a laptop, you don't need to specify a path if you want to use that feed. Just keep in mind that the program is mainly designed for cars on the road.

A simple terminal command to test the script with a video file is:

```sh
python trafficCounter.py -p <path_to_your_video> 
```

There are other parameters that can be modified, but as of now, I have not included a way to change them once the script starts. 

Once the script starts, it will take the first frame of the video and display it. The user must select 4 points on the image to crop it to that size and press q to continue. If you want to use the entire space, then press the enter key and it should continue.

Next, the script will show the cropped image and you can select several points on the screen to create a mask of any shape. The system will only be able to see what is inside that shape. For example:

![Initial cropping](./screenshots/roi_mask_1207.0.jpeg)
![after applying mask](./screenshots/screenshot_1207.0.jpeg)

As you can see, the first image was cropped in a square. The second image shows the result of applying a mask of several points over the image. 

## Blog Post

As part of the research, I wrote blog posts describing the progress made each week. It can be found
at:

https://wordpress.com/posts/andrescscresearch.wordpress.com

## Paper Draft

A paper report of the work is also included in the repository:


[PaperReport_traffic_counter.pdf](./PaperReport_traffic_counter.pdf)

## Other links
I will be posting Python videos to my YoutTube channel, among other projects. If you are interested in checking it out, here is the link:

[Andres Berejnoi Channel](https://www.youtube.com/andresberejnoi)
