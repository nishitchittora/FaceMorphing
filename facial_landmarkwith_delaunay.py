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
ap.add_argument("-i1", "--image1", required=True,
    help="path to input image 1")
ap.add_argument("-i2", "--image2", required=True,
    help="path to input image 2")

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
 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

images = [args["image1"],args["image2"]]
# matrix_points = [[0 for x in range()] for y in range(2)] 
matrix_points=[0 for x in range(2)]
for j,image in enumerate(images):
    image = cv2.imread(image)
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
                    cv2.namedWindow(win_delaunay+"_"+str(j))
                    cv2.moveWindow(win_delaunay+"_"+str(j), 40,30)
                    cv2.imshow(win_delaunay+"_"+str(j), img_copy)
                    cv2.waitKey(1)
         
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
            # cv2.imshow(win_delaunay+"_"+j,image)
            cv2.namedWindow(win_voronoi+"_"+str(j))
            cv2.moveWindow(win_voronoi+"_"+str(j), 800,30)
            cv2.imshow(win_voronoi+"_"+str(j),img_voronoi)
            cv2.waitKey(1)
        matrix_points[j]=points


# Morphing Process starts from here
points = []
alpha = 0.5
for i in xrange(0, len(matrix_points[0])):
    x = ( 1 - alpha ) * matrix_points[0][i][0] + alpha * matrix_points[1][i][0]
    y = ( 1 - alpha ) * matrix_points[0][i][1] + alpha * matrix_points[1][i][1]
    points.append((x,y))

print(points)

img1 = np.float32(cv2.imread(images[0]))
img2 = np.float32(cv2.imread(images[1]))
imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # matrix_points[0] = np.float32(matrix_points[0])
    # matrix_points[1] = np.float32(matrix_points[1])

with open("tri.txt") as file :
    for line in file :
        x,y,z = line.split()
        
        x = int(x)
        y = int(y)
        z = int(z)
        print(x,y,z)
        t1 = [matrix_points[0][x], matrix_points[0][y], matrix_points[0][z]]
        t2 = [matrix_points[1][x], matrix_points[1][y], matrix_points[1][z]]
        t = [points[x], points[y], points[z]]
        print(t1)
        print(t2)
        print(t)
        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)


# Display Result
cv2.imshow("Morphed Face", np.uint8(imgMorph))
cv2.waitKey(0)
