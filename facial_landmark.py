from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, cv2.cv.CV_FILLED, cv2.CV_AA, 0 )
 
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.CV_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.CV_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.CV_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.CV_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.cv.CV_FILLED, cv2.CV_AA, 0)
 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
size = image.shape

rect = (0, 0, size[1], size[0])
subdiv = cv2.Subdiv2D(rect);
animate = True
# detect faces in the grayscale image
rects = detector(gray, 1)
win_delaunay = "Delaunay Triangulation"
win_voronoi = "Voronoi Diagram"
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    points = []


    for (x, y) in shape:
        print(x,y)
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        # x, y = line.split()
        points.append((int(x), int(y)))
 
        # Insert points into subdiv
        for p in points :
            subdiv.insert(p)
             
            # Show animation
            if animate :
                img_copy = image.copy()
                # Draw delaunay triangles
                draw_delaunay( img_copy, subdiv, (255, 255, 255) );
                cv2.namedWindow(win_delaunay)
                cv2.moveWindow(win_delaunay, 40,30)
                cv2.imshow(win_delaunay, img_copy)
                cv2.waitKey(10)
     
        # Draw delaunay triangles
        draw_delaunay( image, subdiv, (255, 255, 255) );
     
        # Draw points
        for p in points :
            draw_point(image, p, (0,0,255))
     
        # Allocate space for Voronoi Diagram
        img_voronoi = np.zeros(image.shape, dtype = image.dtype)
     
        # Draw Voronoi diagram
        draw_voronoi(img_voronoi,subdiv)
     
        # Show results
        # cv2.imshow(win_delaunay,image)
        cv2.namedWindow(win_voronoi)
        cv2.moveWindow(win_voronoi, 800,30)
        cv2.imshow(win_voronoi,img_voronoi)
        cv2.waitKey(10)
 
# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", image)
# cv2.waitKey(0)