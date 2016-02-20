import argparse

##------------Setting up the argument parser
parser = argparse.ArgumentParser(description="""A script to test the OpenCV-based
vehicle counter""")

parser.add_argument('-s','--script',type=file,default='abmGUI.py',help="""The Python script that should be run
with this tester""")


args = vars(parser.parse_args())
###----------------------------------------

dicCount = {'camera_italy.avi':119,'highwayHD.avi':86,
             'highway.avi':86,'autos1HD.avi':27,'autos1.avi':27,
             'autos3HD.avi':29,'autos3.avi':29,'autos4.avi':19,
             'autos4HD.avi':19,'longHighwayHD.avi':3000}

dicVideos = {0:'camera_italy.avi',1:'highwayHD.avi',2:'highway.avi',
             3:'autos1HD.avi',4:'autos1.avi',5:'autos3HD.avi',
             6:'autos3.avi',7:'autos4.avi',8:'autos4HD.avi',9:'longHighwayHD.avi'}

script = args['script']
eval("""script""")


