import argparse 
from traffic_counter import TrafficCounter

def CLI():
    parser = argparse.ArgumentParser(description='Finds the contours on a video file')          #creates a parser object
    parser.add_argument('-p','--path',type=str,help="""A video filename or path.
    Works better with .avi files.
    If no path or name is provided, the camera will be used instead.""")        #instead of using metavar='--path', just type '--path'. For some reason the metavar argument was causing problems
    parser.add_argument('-a','--minArea',type=int,help='The minimum area (in pixels) to draw a bounding box',
                        default=200)
    parser.add_argument('-d','--direction', type=str,default=['H','0.5'],nargs=2,help="""A character: H or V
    representing the orientation of the count line. H is horizontal, V is vertical.
    If not provided, the default is horizontal. The second parameter
    is a float number from 0 to 1 indicating the place at which the
    line should be drawn.""")
    parser.add_argument('-n','--numCount',type=int,default=10,help="""The number of contours to be detected by the program.""")
    parser.add_argument('-w','--webcam',type=int,nargs='+',help="""Allows the user to specify which to use as the video source""")
    parser.add_argument('--rgb',action='store_true',help="Boolean flag to use rbg colors. Default is to use grayscale")
    parser.add_argument('-v','--video_out',type=str,default="",help="Provide a video filename to output")
    parser.add_argument('--video_width',type=int,default=640,help="Videos will be resized to this width. Height will be computed automatically to preserve aspect ratio")
    args = parser.parse_args()
    return args

def get_video_source(video_name):
    try:
        x = int(video_name)
    except:
        x = 5
def main(args):
    video_source   = args.path
    line_direction = args.direction[0]
    line_position  = float(args.direction[1])
    video_width    = args.video_width
    min_area       = int(args.minArea)
    video_out      = False
    numCnts        = int(args.numCount)
    tc = TrafficCounter(video_source,
                        line_direction,
                        line_position,
                        video_width,
                        min_area,
                        video_out,
                        numCnts)

    tc.main_loop()

if __name__ == '__main__':
    args = CLI()
    main(args)