#!/usr/bin/python2
import numpy as np

"""
genmesh.py

This file is responsible for reading the contents of the .msh file 
and creating the mesh mappings for use in DGCUDA
"""

from sys import argv

def genmesh(inFilename, outFilename):
    inFile  = open(inFilename, "rb")
    outFile = open(outFilename, "wb")

    print "Reading file: %s..." % inFilename
    line = inFile.readline()
    while "$Nodes" not in line:
        line = inFile.readline()

    # the next line is the number of vertices
    num_verticies = int(inFile.readline())

    vertex_list = []
    for i in xrange(0,num_verticies):
        s = inFile.readline().split()
        vertex_list.append((float(s[1]), float(s[2])))
    
    # next two lines are just filler
    inFile.readline()
    inFile.readline()

    # next line is the number of elements
    num_elements = int(inFile.readline())

    side_list = []
    elem_list = []
    boundary_list = []
    # add the vertices for each element into elem_list
    print num_elements, "elements..."
    for i in xrange(0,num_elements):

        s = inFile.readline().split()

        # these are sides
        if len(s) == 7:
            boundary = int(s[3])
            v1 = int(s[5]) - 1
            v2 = int(s[6]) - 1
            # store the index of the verticies
            side_list.append((v1, v2))
            boundary_list.append(boundary)

        # and these are elements
        if len(s) == 8:
            v1 = int(s[5]) - 1
            v2 = int(s[6]) - 1
            v3 = int(s[7]) - 1
            # store the index of the verticies
            elem_list.append((v1, v2, v3))

    ##################################################
    # now that we've read in the verticies for the elements and sides,
    # we can begin creating our mesh
    ##################################################

    # number of sides we've added so far
    numsides = 0

    # stores the side number [0, 1, 2] of the left and right element's sides
    left_side_number  = [0] * (num_elements * 3) 
    right_side_number = [0] * (num_elements * 3) 

    # stores the index of the left & right elements
    left_elem  = [0] * (num_elements * 3)
    right_elem = [0] * (num_elements * 3)

    # links to side [0, 1, 2] of the element
    elem_sides = [[0] * 3 for x in xrange(0,num_elements)]

    sidelist = [0] * (num_elements * 3)

    sidemap = {}

    print "Creating mappings..."
    for i, e in enumerate(elem_list):
        # these three vertices define the element
        V1x = vertex_list[e[0]][0]
        V1y = vertex_list[e[0]][1]
        V2x = vertex_list[e[1]][0]
        V2y = vertex_list[e[1]][1]
        V3x = vertex_list[e[2]][0]
        V3y = vertex_list[e[2]][1]

        # determine whether we should add these three sides or not
        sides = [1, 1, 1]

        # look up the hash for each side
        for k, side in enumerate([(e[0], e[1]), (e[1], e[2]), (e[2], e[0])]):
            hash1 = hash(side[:])
            hash2 = hash(side[::-1])
            
            # check to see if (0,1) is a side
            if hash1 in sidemap:
                sides[k] = 0 # this is not a new side
                j = sidemap[hash1]        # get the index of that edge
                right_elem[j] = i         # link that edge to this element
                elem_sides[i][k] = j      # link this element to that edge
                right_side_number[j] = k  # map this edge to the same side

            # check to see if (1,0) is a side
            if hash2 in sidemap:
                sides[k] = 0 # this is not a new side
                j = sidemap[hash2]        # get the index of that edge
                right_elem[j] = i         # link that edge to this element
                elem_sides[i][k] = j      # link this element to that edge
                right_side_number[j] = k  # map this edge to the same side

        """
        for j in xrange(0, numsides):
            # side 1
            if (s1 == 1 and ((side1[j] == e[0] and side2[j] == e[1])
                        or   (side2[j] == e[0] and side1[j] == e[1]))):
                s1 = 0
                # OK, we've added this side to element i
                right_elem[j] = i
                # link the added side j to this element
                elem_s1[i] = j
                right_side_number[j] = 0

        for j in xrange(0, numsides):
            # side 2
            if (s2 == 1 and ((side1[j] == e[1] and side2[j] == e[2])
                        or   (side2[j] == e[1] and side1[j] == e[2]))):
                s2 = 0
                # OK, we've added this side to element i
                right_elem[j] = i
                # link the added side to this element
                elem_s2[i] = j
                right_side_number[j] = 1

        for j in xrange(0, numsides):
            # side 3
            if (s3 == 1 and ((side1[j] == e[0] and side2[j] == e[2])
                        or   (side2[j] == e[0] and side1[j] == e[2]))):
                s3 = 0
                # OK, we've added this side to element i
                right_elem[j] = i
                # link the added side to this element
                elem_s3[i] = j
                right_side_number[j] = 2
        """

        # if we haven't added the side already, add it
        if (sides[0] == 1):
            key = hash((e[0], e[1]))
            sidemap[key] = numsides

            sidelist[numsides] = (e[0], e[1])

            # the side number of this side
            left_side_number[numsides] = 0

            # see if this is a boundary side
            for j, s in enumerate(side_list):
                # side 0 is at this index
                if (s == (e[0], e[1]) or s == (e[1], e[0])):
                    if (boundary_list[j] == 10000):
                        right_elem[numsides] = -1
                    if (boundary_list[j] == 20000):
                        right_elem[numsides] = -2
                    if (boundary_list[j] == 30000):
                        right_elem[numsides] = -3

            # and link the element to this side
            elem_sides[i][0] = numsides

            # make this the left element
            left_elem[numsides] = i
            numsides += 1

        if (sides[1] == 1):
            key = hash((e[1], e[2]))
            sidemap[key] = numsides

            sidelist[numsides] = (e[1], e[2])

            # the side number of this side
            left_side_number[numsides] = 1

            # see if this is a boundary side
            for j, s in enumerate(side_list):
                # side 1 is at this index
                if (s == (e[1], e[2]) or s == (e[2], e[1])):
                    if (boundary_list[j] == 10000):
                        right_elem[numsides] = -1
                    if (boundary_list[j] == 20000):
                        right_elem[numsides] = -2
                    if (boundary_list[j] == 30000):
                        right_elem[numsides] = -3

            # and link the element to this side
            elem_sides[i][1] = numsides

            # make this the left element
            left_elem[numsides] = i
            numsides += 1
            
        if (sides[2] == 1):
            key = hash((e[2], e[0]))
            sidemap[key] = numsides

            sidelist[numsides] = (e[2], e[0])

            # the side number of this side
            left_side_number[numsides] = 2

            # see if this is a boundary side
            for j, s in enumerate(side_list):
                # side 2 is at this index
                if (s == (e[2], e[0]) or s == (e[0], e[2])):
                    if (boundary_list[j] == 10000):
                        right_elem[numsides] = -1
                    if (boundary_list[j] == 20000):
                        right_elem[numsides] = -2
                    if (boundary_list[j] == 30000):
                        right_elem[numsides] = -3

            # and link the element to this side
            elem_sides[i][2] = numsides

            # make this the left element
            left_elem[numsides] = i
            numsides += 1

    print numsides, "sides..."
    print "Sorting mesh..."
    # sort the mesh so that right element = -1 items are first, -2 second, -3 third
    j = 0 # location after the latest right element
    for N in [-1, -2, -3]:
        for i in xrange(0, numsides):
            if right_elem[i] == N:

                # update index for left_elem[j]
                if left_side_number[j] == 0:
                    elem_sides[left_elem[j]][0] = i
                elif left_side_number[j] == 1:
                    elem_sides[left_elem[j]][1] = i
                elif left_side_number[j] == 2:
                    elem_sides[left_elem[j]][2] = i

                # update index for right_elem[j]
                if right_side_number[j] != -1:
                    if right_side_number[j] == 0:
                        elem_sides[right_elem[j]][0] = i
                    elif right_side_number[j] == 1:
                        elem_sides[right_elem[j]][1] = i
                    elif right_side_number[j] == 2:
                        elem_sides[right_elem[j]][2] = i

                # update index for left_elem[i]
                if left_side_number[i] == 0:
                    elem_sides[left_elem[i]][0] = j
                if left_side_number[i] == 1:
                    elem_sides[left_elem[i]][1] = j
                if left_side_number[i] == 2:
                    elem_sides[left_elem[i]][2] = j

                # swap sides i and j
                sidelist[i], sidelist[j] = sidelist[j], sidelist[i]
                left_elem[i] , left_elem[j]  = left_elem[j] , left_elem[i]
                right_elem[i], right_elem[j] = right_elem[j], right_elem[i]
                left_side_number[i] , left_side_number[j]  = left_side_number[j] , left_side_number[i]
                right_side_number[i], right_side_number[j] = right_side_number[j], right_side_number[i]

                # increment j
                j += 1

    elem_s1 = [elem_sides[t][0] for t in xrange(0, num_elements)]
    elem_s2 = [elem_sides[t][1] for t in xrange(0, num_elements)]
    elem_s3 = [elem_sides[t][2] for t in xrange(0, num_elements)]

    # write the mesh to file
    print "Writing file: %s..." % outFilename
    outFile.write(str(len(elem_list)) + "\n")
    for elem, s1, s2, s3 in zip(elem_list, elem_s1, elem_s2, elem_s3):

        V1x = vertex_list[elem[0]][0]
        V1y = vertex_list[elem[0]][1]
        V2x = vertex_list[elem[1]][0]
        V2y = vertex_list[elem[1]][1]
        V3x = vertex_list[elem[2]][0]
        V3y = vertex_list[elem[2]][1]

        # enforce strictly positive jacobian
        J = (V2x - V1x) * (V3y - V1y) - (V3x - V1x) * (V2y - V1y)
        # swap vertex 0 and 1
        if (J < 0):
             V1x, V2x = V2x, V1x
             V1y, V2y = V2y, V1y

        outFile.write("%.015lf %.015lf %.015lf %.015lf %.015lf %.015lf %i %i %i\n" % 
                                                             (V1x, V1y, V2x, V2y, V3x, V3y,
                                                              s1, s2, s3))

    outFile.write(str(numsides) + "\n")
    for i in xrange(0, numsides):
        outFile.write("%.015lf %.015lf %.015lf %.015lf %i %i %i %i\n" % 
                                        (vertex_list[sidelist[i][0]][0], vertex_list[sidelist[i][0]][1],
                                         vertex_list[sidelist[i][1]][0], vertex_list[sidelist[i][1]][1],
                                         left_elem[i], right_elem[i], 
                                         left_side_number[i], right_side_number[i]))

    outFile.close()

if __name__ == "__main__":
    try:
        inFilename  = argv[1] 
        outFilename = argv[2]
        genmesh(inFilename, outFilename)
    except Exception:
        print "Usage: genmesh [infile] [outfile]"

