/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef NN_GRAPH_2D_GEOMETRY_H
#define NN_GRAPH_2D_GEOMETRY_H

#include <math.h>

#define M_PI 3.14159265359

typedef struct {
    float x;
    float y;
} point;

typedef struct {
    float A;
    float B;
    float C;
    float slope;
    float y_intercept;
} line;

typedef struct {
    line line;
    point start;
    point end;
} line_segment;

typedef struct {
    point vertices[4];
    line_segment edges[4];
    float height;
	float width;
    float left_bound;
    float right_bound;
    float top_bound;
    float bottom_bound;
	float angle;
} rectangle;


// creates a line that passes through the given pair of points
line create_line(point p1, point p2){
    line result;

    if(p1.x == p2.x){
        result.A = 1;
        result.B = 0;
        result.C = -p1.x;
    }
    else{
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float m = dy / dx;
        float b = p1.y - m * p1.x;

        if(dy > 0){
            result.A = dy;
            result.B = dx;
            result.C = dx * b;
        }
        else{
            result.A = -dy;
            result.B = -dx;
            result.C = -dx * b;
        }
        result.slope = m;
        result.y_intercept = b;
    }

    return result;
}

// creates a line segment that starts and ends with the given points
line_segment create_line_segment(point p1, point p2){
    line_segment result;

    result.line = create_line(p1, p2);

    if(p1.x == p2.x){			// line is vertical
        // make start the lower of the two points
        if(p1.y < p2.y){
            result.start = p1;
            result.end = p2;
        }
        else{
            result.start = p2;
            result.end = p1;
        }
    }
        // make start the point to the left
    else if(p1.x < p2.x){
        result.start = p1;
        result.end = p2;
    }
    else{
        result.start = p2;
        result.end = p1;
    }

    return result;
}

// creates a rectangle with the given vertices
// the vertices are expected to be given in counter-clockwise sequence
rectangle create_rectangle(point p1, point p2, point p3, point p4){
    rectangle result;

    result.vertices[0] = p1;
    result.vertices[1] = p2;
    result.vertices[2] = p3;
    result.vertices[3] = p4;

    result.edges[0] = create_line_segment(p1, p2);
    result.edges[1] = create_line_segment(p2, p3);
    result.edges[2] = create_line_segment(p3, p4);
    result.edges[3] = create_line_segment(p4, p1);

    result.top_bound = p1.y;
    result.bottom_bound = p1.y;
    result.left_bound = p1.x;
    result.right_bound = p1.x;

    int i;
    for(i = 0; i < 4; i++){
        if(result.vertices[i].x < result.left_bound) { result.left_bound = result.vertices[i].x; }
        if(result.vertices[i].x > result.right_bound) { result.right_bound = result.vertices[i].x; }
        if(result.vertices[i].y < result.bottom_bound) { result.bottom_bound = result.vertices[i].y; }
        if(result.vertices[i].y > result.top_bound) { result.top_bound = result.vertices[i].y; }
    }

    return result;
}

// creates a rectangle with the lower left vertex at p1 and upper right vertex at p2
rectangle create_upright_rectangle(float* rectangle_data){
    point p1, p2, p3, p4;

    p1.x = rectangle_data[2]; p1.y = rectangle_data[3];
    p2.x = rectangle_data[0]; p2.y = rectangle_data[3];
    p3.x = rectangle_data[0]; p3.y = rectangle_data[1];
    p4.x = rectangle_data[2]; p4.y = rectangle_data[1];

    rectangle result = create_rectangle(p1, p2, p3, p4);
	
	result.height = rectangle_data[3] - rectangle_data[1];
	result.width = rectangle_data[2] - rectangle_data[0];
	result.angle = 0;
	
	return result;
}

// creates a rectangle given rotated rectangle data in the format:
// x_center, y_center, width, height, angle
rectangle create_rotated_rectangle(float *rectangle_data){

    float x = rectangle_data[2]*0.5f;
    float y = rectangle_data[3]*0.5f;
    float rotation_angle = rectangle_data[4] * (float)(M_PI / 180.f);
    float hyp = hypotf(x,y);
    float angle1 = atan2f(y,x);
    float angle2 = angle1 - rotation_angle;
    angle1 += rotation_angle;
    point vertices[4];

    // rotate around the center
    vertices[0].x =  hyp * cosf(angle1);   vertices[0].y = hyp * sinf(angle1);
    vertices[1].x = -hyp * cosf(angle2);   vertices[1].y = hyp * sinf(angle2);
    vertices[2].x = -vertices[0].x;       vertices[2].y = -vertices[0].y;
    vertices[3].x = -vertices[1].x;       vertices[3].y = -vertices[1].y;

    // translate
    vertices[0].x += rectangle_data[0]; vertices[0].y += rectangle_data[1];
    vertices[1].x += rectangle_data[0]; vertices[1].y += rectangle_data[1];
    vertices[2].x += rectangle_data[0]; vertices[2].y += rectangle_data[1];
    vertices[3].x += rectangle_data[0]; vertices[3].y += rectangle_data[1];

    rectangle result = create_rectangle(vertices[0], vertices[1], vertices[2], vertices[3]);
	
	result.height = rectangle_data[3];
	result.width = rectangle_data[2];
	result.angle = rectangle_data[4];
	
	return result;
}

// returns the number of times the given two line segments intersect and populates
// the intersections parameter with the intersection points
int line_segment_intersections(line_segment segment1, line_segment segment2, point* intersections){

    // both line segments are vertical lines
    if(segment1.line.B == 0 && segment2.line.B == 0){

        // line segments are not on the same line, no intersection
        if(segment1.start.x != segment2.start.x){
            return 0;
        }

        // line segments are on the same line but don't overlap, no intersection
        if(segment1.start.y > segment2.end.y ||
           segment2.start.y > segment1.end.y){
            return 0;
        }

        // line segments are on the same line and overlap, there are two intersections
        int count = 0;
        if(segment1.start.y >= segment2.start.y && segment1.start.y <= segment2.end.y){
            intersections[count] = segment1.start;
            count++;
        }
        if(segment1.end.y >= segment2.start.y && segment1.end.y <= segment2.end.y){
            intersections[count] = segment1.end;
            count++;
        }
        if(count < 2 && segment2.start.y >= segment1.start.y && segment2.start.y <= segment1.end.y){
            intersections[count] = segment2.start;
            count++;
        }
        if(count < 2 && segment2.end.y >= segment1.start.y && segment2.end.y <= segment1.end.y){
            intersections[count] = segment2.end;
            count++;
        }

        return 2;
    }

    // one line segment is vertical
    if(segment1.line.B == 0 || segment2.line.B == 0){

        // make segment1 the vertical one for convenience
        if(segment2.line.B == 0){
            line_segment tmp = segment1;
            segment1 = segment2;
            segment2 = tmp;
        }

        // line segments don't cross, no intersection
        if(segment2.start.x > segment1.start.x || segment2.end.x < segment1.start.x){
            return 0;
        }
        float int_y = segment2.line.slope * segment1.start.x + segment2.line.y_intercept;
        if(int_y < segment1.start.y || int_y > segment1.end.y){
            return 0;
        }

        intersections[0].x = segment1.start.x;
        intersections[0].y = segment2.line.slope * segment1.start.x + segment2.line.y_intercept;

        return 1;
    }

    // neither line segment is vertical and they are parallel to each other
    if(segment1.line.slope == segment2.line.slope){
        // line segments are parallel but not on the same line, no intersections
        if(segment1.line.y_intercept != segment2.line.y_intercept){
            return 0;
        }

        // line intersections are on the same line and overlap, there are two intersections
        int count = 0;
        if(segment1.start.x >= segment2.start.x && segment1.start.x <= segment2.end.x){
            intersections[count] = segment1.start;
            count++;
        }
        if(segment1.end.x >= segment2.start.x && segment1.end.x <= segment2.end.x){
            intersections[count] = segment1.end;
            count++;
        }
        if(count < 2 && segment2.start.x >= segment1.start.x && segment2.start.x <= segment1.end.x){
            intersections[count] = segment2.start;
            count++;
        }
        if(count < 2 && segment2.end.x >= segment1.start.x && segment2.end.x <= segment1.end.x){
            intersections[count] = segment2.end;
            count++;
        }

        return 2;
    }

    // intercept point for the two lines
    float int_x = (segment2.line.y_intercept - segment1.line.y_intercept) / (segment1.line.slope - segment2.line.slope);

    // if the intersect point is within the line segments then it's a valid intersection
    if(segment1.start.x <= int_x && int_x <= segment1.end.x && segment2.start.x <= int_x && int_x <= segment2.end.x){
        intersections[0].x = int_x;
        intersections[0].y = segment1.line.slope * int_x + segment1.line.y_intercept;
        return 1;
    }
    else{
        return 0;
    }
}

// returns 1 if point p is inside rectanlge r, 0 otherwise
int is_point_inside_rectangle(point p, rectangle r){

    // if the point is out of bounds then it is not inside the rectangle
    if(p.x < r.left_bound || p.x > r.right_bound ||
       p.y < r.bottom_bound || p.y > r.top_bound){
        return 0;
    }

    // create a line segment from p to some point outside the rectangle bounds
    point test_point;
    test_point.x = r.right_bound + 1.f;
    test_point.y = p.y;
    line_segment test_segment = create_line_segment(p, test_point);

    // count the number of times the test segment intersects the edges of the rectangle
    int intersections = 0;
    point intersect_points[8];
    int i;
    for(i = 0; i < 4; i++){
        intersections += line_segment_intersections(test_segment, r.edges[i], intersect_points);
    }

    // if the line segment intersects the rectangle odd number of times then the point is inside the rectangle
    if(intersections % 2 == 0){
        return 0;
    }
    else{
        return 1;
    }
}

// compare function for sorting points on the x value
static int compare_points_x (const void * a, const void * b) {
    point pa = *(point*)a;
    point pb = *(point*)b;
    return ( pb.x - pa.x >= 0.0 ? -1 : 1 );
}

// returns the number of vertices that form the intersection between the rectangles r1 and r2 and populates
// the 'vertices' parameter with those vertices in counter-clockwise sequence
int get_rectangle_intersection(rectangle r1, rectangle r2, point *vertices){

    point unordered_points[8];
    point current_intersections[2];
    int i, j, k, total_vertices = 0;

    // detect all edge intersections
    for(i = 0; i < 4; i++){
        for(j = 0; j < 4; j++){
            int num_intersections = line_segment_intersections(r1.edges[i], r2.edges[j], current_intersections);
            for(k = 0; k < num_intersections; k++){
                unordered_points[total_vertices] = current_intersections[k];
                total_vertices++;
            }
        }
    }

    // check for rectangle 1 vertices inside rectangle 2
    for(i = 0; i < 4; i++){
        if(is_point_inside_rectangle(r1.vertices[i], r2)){
            unordered_points[total_vertices] = r1.vertices[i];
            total_vertices++;
        }
    }

    // check for rectangle 2 vertices inside rectangle 1
    for(i = 0; i < 4; i++){
        if(is_point_inside_rectangle(r2.vertices[i], r1)){
            unordered_points[total_vertices] = r2.vertices[i];
            total_vertices++;
        }
    }

    // if there are 2 or fewer vertices then just return them
    if(total_vertices <= 2){
        for(i = 0; i < total_vertices; i++){
            vertices[i] = unordered_points[i];
        }
        return total_vertices;
    }

    // we now need to order the vertices in counter-clockwise order

    // find the leftmost and rightmost vertices
    // break ties by favouring lower y values for leftmost and higher y values for rightmost
    int leftmost = 0, rightmost = 0;
    for(i = 0; i < total_vertices; i++){
        if(unordered_points[leftmost].x > unordered_points[i].x ||
           (unordered_points[leftmost].x == unordered_points[i].x && unordered_points[leftmost].y > unordered_points[i].y)){
            leftmost = i;
        }
        if(unordered_points[rightmost].x < unordered_points[i].x ||
           (unordered_points[rightmost].x == unordered_points[i].x && unordered_points[rightmost].y < unordered_points[i].y)){
            rightmost = i;
        }
    }

    // draw a line from the leftmost vertex to the righmost vertex
    line middle = create_line(unordered_points[leftmost], unordered_points[rightmost]);

    // identify vertices that are above and below the line
    point above[8];
    point below[8];
    int above_count = 0, below_count = 0;
    for(i = 0; i < total_vertices; i++){
        if(i == leftmost || i == rightmost){
            continue;
        }

        if(middle.slope * unordered_points[i].x + middle.y_intercept > unordered_points[i].y){
            below[below_count] = unordered_points[i];
            below_count++;
        }
        else{
            above[above_count] = unordered_points[i];
            above_count++;
        }
    }

    // sort vertices above and below the line by their x values
    qsort(above, above_count, sizeof(point), compare_points_x);
    qsort(below, below_count, sizeof(point), compare_points_x);

    // add the vertices to the final result in this order:
    // 1) leftmost vertex
    // 2) vertices below the line by increasing x values
    // 3) rightmost vertex
    // 4) vertices above the line by decreasing x values
    vertices[0] = unordered_points[leftmost];
    total_vertices = 1;
    for(i = 0; i < below_count; i++){
        vertices[total_vertices] = below[i];
        total_vertices++;
    }
    vertices[total_vertices] = unordered_points[rightmost];
    total_vertices++;
    for(i = 0; i < above_count; i++){
        vertices[total_vertices] = above[above_count-i-1];
        total_vertices++;
    }

    return total_vertices;
}

// returns the area of the polygon formed by the given vertices
// the vertices are assumed to be given in counter-clockwise order
// does not work with self-intersecting polygons
float get_polygon_area(point *vertices, int num_vertices){

    if(num_vertices <= 2){
        return 0.f;
    }

    float numerator = 0.f;
    int i;
    for(i = 0; i < num_vertices-1; i++){
        numerator += vertices[i].x * vertices[i+1].y - vertices[i].y * vertices[i+1].x;
    }
    numerator += vertices[num_vertices-1].x * vertices[0].y - vertices[num_vertices-1].y * vertices[0].x;

    return fabsf(numerator/2.f);
}

// returns the area of the intersection between rectangles r1 and r2
float get_rotated_rectangles_intersection_area(rectangle r1, rectangle r2){

    if(r1.right_bound < r2.left_bound ||
       r1.left_bound > r2.right_bound ||
       r1.top_bound < r2.bottom_bound ||
       r1.bottom_bound > r2.top_bound){
        return 0.f;
    }

    point intersection_vertices[8];
    int num_vertices = get_rectangle_intersection(r1, r2, intersection_vertices);

    if(num_vertices <= 2){
        return 0.f;
    }

    return get_polygon_area(intersection_vertices, num_vertices);
}

// returns the area of the intersection between rectangles r1 and r2, assuming that both of them are upright
float get_upright_rectangles_intersection_area(rectangle r1, rectangle r2){

    if(r1.right_bound < r2.left_bound ||
       r1.left_bound > r2.right_bound ||
       r1.top_bound < r2.bottom_bound ||
       r1.bottom_bound > r2.top_bound){
        return 0.f;
    }

    float xx1 = r1.left_bound > r2.left_bound     ? r1.left_bound   : r2.left_bound;
    float yy1 = r1.bottom_bound > r2.bottom_bound ? r1.bottom_bound : r2.bottom_bound;
    float xx2 = r1.right_bound > r2.right_bound   ? r2.right_bound  : r1.right_bound;
    float yy2 = r1.top_bound > r2.top_bound       ? r2.top_bound    : r1.top_bound;

    float w = xx2 - xx1 + 1.0f;
    float h = yy2 - yy1 + 1.0f;

    w = w < 0.0 ? 0.0 : w;
    h = h < 0.0 ? 0.0 : h;

    return w * h;
}

float get_rectangles_intersection_area(rectangle r1, rectangle r2){
	if(r1.angle == 0 && r2.angle == 0){
		return get_upright_rectangles_intersection_area(r1, r2);
	}
	else{
		return get_rotated_rectangles_intersection_area(r1, r2);
	}
}


#endif //NN_GRAPH_2D_GEOMETRY_H
