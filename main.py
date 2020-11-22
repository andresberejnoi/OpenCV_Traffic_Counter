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
    parser.add_argument('-vo','--video_out',type=str,default="",help="Provide a video filename to output")
    parser.add_argument('-vw','--video_width',type=int,default=640,help="Videos will be resized to this width. Height will be computed automatically to preserve aspect ratio")
    parser.add_argument('-vp','--video_params',type=str,default=['mjpg','avi'],nargs=2,help="Provide video codec and extension (in that order) for the output video. Example: `--video_params mjpg avi`. Default values are mjpg and avi")
    parser.add_argument('-sf','--starting_frame',type=int,default=10,help="Select the starting frame for video analysis. All frames before that will still be used for the background average")
    args = parser.parse_args()
    return args

def make_video_params_dict(video_params):
    codec     = video_params[0]
    extension = video_params[1]
    
    params_dict = {
        'codec'    :codec,
        'extension':extension,
    }
    return params_dict

def main(args):
    video_source   = args.path
    line_direction = args.direction[0]
    line_position  = float(args.direction[1])
    video_width    = args.video_width
    min_area       = int(args.minArea)
    video_out      = args.video_out
    numCnts        = int(args.numCount)
    video_params   = make_video_params_dict(args.video_params)
    starting_frame = args.starting_frame
    tc = TrafficCounter(video_source,
                        line_direction,
                        line_position,
                        video_width,
                        min_area,
                        video_out,
                        numCnts,
                        video_params,
                        starting_frame,)

    tc.main_loop()

if __name__ == '__main__':
    args = CLI()
    main(args)