import cv2
import numpy as np
import argparse
from imutils import face_utils
import imutils
import random


def readPoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image,(x,y),2,(255,0,0),-1)
        mouse.append((x,y))
        # image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = image.shape

        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect);
        animate = True
        # detect faces in the grayscale image
        win_delaunay = "Delaunay Triangulation"
        win_voronoi = "Voronoi Diagram"
        # for (i, rect) in enumerate(rects):

        # points.append((int(x), int(y))) 
        # Insert points into subdiv
        for p in mouse:
            print(p)
            subdiv.insert(p)
            # Show animation
            if animate :
                img_copy = image.copy()
                # Draw delaunay triangles
                draw_delaunay( img_copy, subdiv, (255, 255, 255) );
                # cv2.namedWindow(win_delaunay)
                # cv2.moveWindow(win_delaunay, 40,30)
                # cv2.imshow(win_delaunay, img_copy)
                # cv2.waitKey(10)

        # Draw delaunay triangles
        draw_delaunay( image, subdiv, (255, 255, 255) );

        # Draw points
        for p in mouse :
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

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#     help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
#     help="path to input image")
# args = vars(ap.parse_args())
mouse = []
# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
image = cv2.imread('4.jpg')
cv2.setMouseCallback('image',draw_circle)

# image = cv2.imread(args["image"])


while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print mouse
