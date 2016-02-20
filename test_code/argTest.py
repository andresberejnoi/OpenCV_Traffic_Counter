import argparse

ap = argparse.ArgumentParser(description='Test the argparse module')

ap.add_argument('-t','--text',type=str,help='text to print',default='hello')
ap.add_argument('-p','--fileName', type=str,default='/home',help="""File name or absolute path\nto file""")
ap.add_argument('-l','--list',type=str,nargs=2,default=['test','0,2'],help='Testing argument of several parameters')

                


args = vars(ap.parse_args())
if args.get('text',None) is None and args.get('fileName',None) is None:
    print ('No arguments provided')
    exit()

print('len(list): '+ str(len(args['list'])))
print('list: ' + str(args['list']))
print(args.get('text'))
print(args.get('fileName'))

                             
