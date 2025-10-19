from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import texture
import re
from PIL import Image
from collections import Counter
import colorsys
import random
import glm
from box3 import Box3
from ctypes import c_uint32, cast, POINTER
import time
from metrics import erode_mask 

masking_shader_str = """
#version 460

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in; 

uniform  int resolution_width;
uniform  int resolution_height;

struct MaskInfo {
    ivec4 index;  // x: id, y:index, z: size_x, w: size_y
    ivec2 corner;     // top-left or other custom meaning
};

layout(std430, binding = 0) buffer bIndexToMasks {
    MaskInfo indexToMasks[]; // indexToMasks[i].index.x = index to the mask i, indexToMasks[i].y = size x of the mask i, indexToMasks[i].z = size y of the mask i
};

layout(std430, binding = 1) buffer bMasks {
    uint masks[]; // masks[i] = 1/0 if the corresponding pixel is on in the mask. THis buffer contains all the masks
};


uniform int uMaskSize;
uniform mat4 uViewCam;                                  // camera view matrix
layout(binding = 12)  uniform sampler2D uColorTexture;   // for each pixel of the image, the coordinates in parametric space
uniform float uRangeThreshold;                          // threshold for the range of the mask


layout(binding = 13) uniform sampler2D uTriangleMap;  // for each pixel of the image, to which triangle it belongs


// input/output image

layout(std430, binding = 2) buffer bCoverage {
    uint coverage_ssbo[]; // masks[i*sizemask+j] = onto which triangles the sample j of mask i falls
};

layout(std430, binding = 6) buffer bCoverageWeight {
    float coverage_weight_ssbo[]; // weight of the sample
};

layout(std430, binding = 3) buffer bVerts {
    float verts[]; // mesh vertices
};

layout(std430, binding = 4) buffer bFaces {
    uint faces[]; // mesh faces
};

layout(std430, binding = 5) buffer bAvgCol {
    vec4 avg_col[]; // avg_col[i] = average color
};

void main() {

    const int MAX_ADJ = 128;

    // get the mask id
    uint id = gl_GlobalInvocationID.x;

    int n_samples = 0;

    int idMask = indexToMasks[id].index.x;    // starting position of the mask in the buffer
    int offset = indexToMasks[id].index.y;    // starting position of the mask in the buffer
    int width  = indexToMasks[id].index.z;    // width of the mask
    int height = indexToMasks[id].index.w;    // height of the mask

    ivec2 corner = indexToMasks[id].corner;   // top-left corner of the mask in the image
    
    float max_z = -1000.0;
    float min_z = 1000.0;
    // ................................................
    
    vec3 sum_col = vec3(0.0,0.0,0.0);
    float total_cov_w = 0.f;
    for(int i = 0; i < width; i++) 
        for(int j = 0; j < height; j++) {
            int ii = corner.x + i;
            int jj = corner.y + height - 1 - j;

            vec2 uv = vec2(float(ii)/float(resolution_width), 1.0 - float(jj)/float(resolution_height));
            uint v =  masks[offset + i + (height-1-j) * width];

            if(v > 0)
            {
               sum_col +=  texture(uColorTexture, uv).xyz;

               // read the index to the triangle
               int idTri =  int(texelFetch(uTriangleMap, ivec2(ii, resolution_height - jj), 0).r);

                n_samples++;   
                coverage_ssbo[id*uMaskSize + n_samples] = idTri;

                float x = (i-width/2.f)/(width/2.f);
                float y = (j-height/2.f)/(height/2.f);
                float w = exp(-0.8512 * (x*x+y*y) ) ; 
                coverage_weight_ssbo[id*uMaskSize + n_samples] = w ; 
                total_cov_w += w;
                
                // read the vertex of the triangle
                vec3 p = vec3(verts[faces[idTri*3]*3], verts[faces[idTri*3]*3+1], verts[faces[idTri*3]*3+2]);

                     
                vec4 p0 = uViewCam * vec4(p, 1.0);
        
                if(p0.z > max_z) max_z = p0.z;
                if(p0.z < min_z) min_z = p0.z;
            }
        } 
      coverage_ssbo[id*uMaskSize ]          = n_samples;
      coverage_weight_ssbo[id*uMaskSize ]   = total_cov_w;
      
      if(abs(max_z-min_z) > uRangeThreshold) 
        coverage_ssbo[id*uMaskSize ] = 0;   
      avg_col[id] = vec4(sum_col *1.0/float(n_samples),0.0);  
}
"""
range_shader_str = """
#version 460

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in; 

uniform  int resolution_width;
uniform  int resolution_height;

struct MaskInfo {
    ivec4 index;  // x: id, y:index, z: size_x, w: size_y
    ivec2 corner;     // top-left or other custom meaning
};

layout(std430, binding = 0) buffer bIndexToMasks {
    MaskInfo indexToMasks[]; // indexToMasks[i].index.x = index to the mask i, indexToMasks[i].y = size x of the mask i, indexToMasks[i].z = size y of the mask i
};

layout(std430, binding = 1) buffer bMasks {
    uint masks[]; // masks[i] = 1/0 if the corresponding pixel is on in the mask. THis buffer contains all the masks
};



uniform int uMaskSize;
uniform mat4 uViewCam;              // camera view matrix

layout(binding = 13) uniform sampler2D uTriangleMap;  // for each pixel of the image, to which triangle it belongs
uniform int uNMasks;


layout(std430, binding = 2) buffer bCoverage {
    uint coverage_ssbo[]; // masks[i*sizemask+j] = onto which triangles the sample j of mask i falls
};

layout(std430, binding = 3) buffer bVerts {
    float verts[]; // mesh vertices
};

layout(std430, binding = 4) buffer bFaces {
    uint faces[]; // mesh faces
};

layout(std430, binding = 5) buffer bRange {
    float range[]; 
};

void main() {


    // get the mask id
    uint id = gl_GlobalInvocationID.x;

    if(id > uNMasks - 1)
    return;

    int idMask = indexToMasks[id].index.x;    // starting position of the mask in the buffer
    int offset = indexToMasks[id].index.y;    // starting position of the mask in the buffer
    int width  = indexToMasks[id].index.z;    // width of the mask
    int height = indexToMasks[id].index.w;    // height of the mask

    ivec2 corner = indexToMasks[id].corner;   // top-left corner of the mask in the image
    
    float max_z = -1000.0;
    float min_z = 1000.0;
    // ................................................
    
    vec3 sum_col = vec3(0.0,0.0,0.0);
    for(int i = 0; i < width; i++) 
        for(int j = 0; j < height; j++) {
            int ii = corner.x + i;
            int jj = corner.y + j;

            uint v =  masks[offset + i + (height-1-j) * width];
            
            if(v > 0)
            {

               // read the index to the triangle
               int idTri =  int(texelFetch(uTriangleMap, ivec2(ii, resolution_height - jj), 0).r);
                
                // read the vertex of the triangle
                vec3 p = vec3(verts[faces[idTri*3]*3], verts[faces[idTri*3]*3+1], verts[faces[idTri*3]*3+2]);
                     
                vec4 p0 = uViewCam * vec4(p, 1.0);
        
                if(p0.z > max_z) max_z = p0.z;
                if(p0.z < min_z) min_z = p0.z;

                
            }
        } 
    
    range[id] = abs(max_z - min_z);   
}
"""

fluo_shader_str = """
#version 460

layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in; 



// for debug
// layout(binding = 0, rgba32f) writeonly uniform image2D _dbg_readed;

struct MaskInfo {
    ivec4 index;  // x: id, y:index, z: size_x, w: size_y
    ivec2 corner; // top-left or other custom meaning
};

layout(std430, binding = 0) buffer bIndexToMasks {
    MaskInfo indexToMasks[]; // indexToMasks[i].index.x = index to the mask i, indexToMasks[i].y = size x of the mask i, indexToMasks[i].z = size y of the mask i
};

layout(std430, binding = 1) buffer bMasks {
    uint masks[]; 
};

layout(binding = 14)  uniform sampler2D uColorTexture1;  
layout(binding = 15)  uniform sampler2D uColorTexture2;  
layout(binding = 16)  uniform sampler2D uPosTexture; 


layout(std430, binding = 5) buffer bAvgCol {
    vec4 avg_col[]; // avg_col[i] = average color
};

uniform  int resolution_width_rgb;
uniform  int resolution_height_rgb;

uniform mat4 uViewCam_FLUO;                                  // camera view matrix
uniform  int resolution_width;
uniform  int resolution_height;

 // Properties
uniform    float pixel_width;
uniform    float pixel_height;
uniform    float focal_length;

// Calibration
uniform    float f;
uniform    float cx; // this is the offset w.r.t. the center
uniform    float cy; // this is the offset w.r.t. the center
uniform    float k1;
uniform    float k2;
uniform    float k3;
uniform    float p1;
uniform    float p2;


vec2 xyz_to_uv(vec3 p){
    float x = p.x/p.z;
    float y = -p.y/p.z;
    float r = sqrt(x*x+y*y);
    float r2 = r*r;
    float r4 = r2*r2;
    float r6 = r4*r2;
    float r8 = r6*r2;

    float A = (1.0+k1*r2+k2*r4+k3*r6  /*+k4*r8*/ ); 
    float B = (1.0 /* +p3*r2+p4*r4 */ );

    float xp = x * A+ (p1*(r2+2*x*x)+2*p2*x*y) * B;
    float yp = y * A+ (p2*(r2+2*y*y)+2*p1*x*y) * B;

    float u = resolution_width*0.5+cx+xp*f; //+xp*b1+yp*b2
    float v = resolution_height*0.5+cy+yp*f;

    u /= resolution_width;
    v /= resolution_height;

    return vec2(u,v);
}

void main() {

    // get the mask id
    uint id = 0;

    int n_samples = 0;

    ivec2 corner = indexToMasks[id].corner;   // top-left corner of the mask in the image
    
    int idMask = indexToMasks[id].index.x;    // starting position of the mask in the buffer
    int offset = indexToMasks[id].index.y;    // starting position of the mask in the buffer
    int width  = indexToMasks[id].index.z;    // width of the mask
    int height = indexToMasks[id].index.w;    // height of the mask
    // ................................................
    
    vec3 sum_col = vec3(0.0,0.0,0.0);
    vec3 pos;
    vec3 posVS ;
    vec2 uv;
    for(int i = 0; i < width; i++) 
        for(int j = 0; j < height; j++) {
            int ii = corner.x + i;
            int jj = corner.y + height - 1 - j;

            uint v =  masks[offset + i + (height-1-j) * width];
            
            if(v > 0){ 
                uv = vec2(float(ii)/float(resolution_width_rgb), 1.0 - float(jj)/float(resolution_height_rgb));

                pos = texture(uPosTexture, uv).xyz;
                posVS = (uViewCam_FLUO*vec4(pos,1.0)).xyz;
                uv = xyz_to_uv( posVS );
                
                vec3 col =  texture(uColorTexture1, uv).xyz- texture(uColorTexture2, uv).xyz;
                sum_col += col;

                n_samples++;

                //imageStore(_dbg_readed, ivec2(uv*vec2(resolution_width,resolution_height)), vec4(1,1,1,1.0));
 
            }
        }  
    avg_col[id] = vec4(sum_col *1.0/float(n_samples),n_samples);  

    //for (int i = 100; i < 200; i++)
    //    for (int j = 100; j < 200; j++)
    //        imageStore(_dbg_readed, ivec2(i, j), vec4(0, 1, 0, 1.0));
   
}
"""

class cshader:
    def __init__(self, shader_str):
        self.uniforms = {}
        self.program =compileProgram(compileShader(shader_str, GL_COMPUTE_SHADER))

    def uni(self, name):
            if name not in self.uniforms:
                self.uniforms[name] = glGetUniformLocation(self.program, name)
            return self.uniforms[name]
    


def setup_cshader( ):
    global program_mask
    global triangle_map_texture
    global range_shader
    global fluo_shader
    global coverage
    global vertices
    global faces 
    
    program_mask = cshader(masking_shader_str)
    range_shader = cshader(range_shader_str)
   

    glActiveTexture(GL_TEXTURE13)
    glBindTexture(GL_TEXTURE_2D, triangle_map_texture)

    # to do it once
     
    verts_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, verts_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, vertices.flatten().astype(np.float32).nbytes, vertices.flatten().astype(np.float32), GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, verts_ssbo)
 
    faces_ssbo = glGenBuffers(1)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, faces_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, faces.flatten().nbytes, faces.flatten(), GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, faces_ssbo)



def random_color():
    # Generate a random hue value
    h = random.random()
    # Saturation and value are set to 1 for a fully saturated color
    s = 1.0
    v = 1.0
    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Convert to 0-255 range
    return [int(r * 255), int(g * 255), int(b * 255)]

def color_the_domain(m,v_color,cov_thr = 0 ):
    
    for el in m.domain_mask:
        domain_mask[el[0],el[1]]= v_color

    if sum(m.triangles.values()) / m.ones_w> cov_thr:
#    if  float(len(m.triangles)) / m.n_triangles  > 0.5:  
        for id, count  in m.triangles.items():
            c0 = id*3*3
            tri_color[c0:c0+3]  =   [color/255.0 for color in v_color]
            tri_color[c0+3:c0+6] =  [color/255.0 for color in v_color]
            tri_color[c0+6:c0+9] =  [color/255.0 for color in v_color]


class mask:
     def __init__(self, name, id_texture, img_data,img_name,id_camera, w,h, X, Y, C):
            self.filename = name
            self.id_texture = id_texture
            self.img_name = img_name
            self.img_data = img_data
            self.id_camera = id_camera
            self.w = w
            self.h = h
            self.X = X
            self.Y = Y
            self.C = C
            self.ones = 0
            self.domain_mask = []
            self.triangles = []
            self.bbox = Box3()

    
class node:
    def __init__(self, mask):
        self.mask = mask
        self.color = random_color()
        self.adjacents = []

    def add_adj(self, id_adj, weight):
        self.adjacents.append(arc(id_adj,weight))

class arc:
    def __init__(self, id_node, weight):
        self.id_node = id_node
        self.weight = weight

class mask_graph:
    def __init__(self):
        self.nodes = []
        self.connected_components = []
    
    def add_node(self, n):
        self.nodes.append(n)

    def add_edge(self,id0,id1,weight):
        self.nodes[id0].add_adj(id1,weight)
        self.nodes[id1].add_adj(id0,weight)

    def count_connected_components(self):
        #self.weight_arcs()
        self.connected_components = []
        visited = set()
        #node_indices = list(range(len(self.nodes)))
        #self.complete_clicque(node_indices)
       
        def connected_component(start_index, id, threshold=0.0): 
            stack = [start_index]
            class Component(list):
                pass
            component = Component()
            component.v_color = random_color()
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    component.append(curr)
                    self.nodes[curr].mask.id_comp = id
                    color_the_domain(self.nodes[curr].mask,component.v_color)
                    stack.extend(adj.id_node for adj in self.nodes[curr].adjacents if adj.weight > threshold)  # Add unvisited neighbors with weight > 0.7
            return component

        def max_weight(component):
            max_weight = -1
            max_node = -1
            for node in component:
                if self.nodes[node].mask.ones > max_weight:
                    max_weight = self.nodes[node].mask.ones
                    max_node = node
            print (f"Max weight: {max_weight}")        
            return max_weight
        
        n_components = 0
        for i in range(len(self.nodes)):
            if i not in visited:
                component  = connected_component(i,len(self.connected_components),0.5)
                #self.complete_clicque(component)
                #component_final  = connected_component(i)
                #if max_weight(component) > 1600:# to be replaced with the 80 percentile
                self.connected_components.append(component)
                n_components+=1

        return n_components

    def weight_arcs(self):
        for node in self.nodes:
            for adj in node.adjacents:
                overlap = node.mask.triangles & self.nodes[adj.id_node].mask.triangles
                weight = sum(overlap.values())
                adj.weight = weight/node.mask.ones

    def complete_clicque(self, inodes):
        for nodea in inodes:
            for nodeb in inodes:
                if nodeb is not nodea and nodeb not in [adj.id_node for adj in self.nodes[nodea].adjacents]:
                    weight = sum((self.nodes[nodea].mask.triangles & self.nodes[nodeb].mask.triangles).values())/self.nodes[nodea].mask.ones
                    if weight > 0.0:
                        self.nodes[nodea].add_adj(nodeb, weight)

    def test_redundancy(self, id_comp):
        node = self.nodes[self.connected_components[id_comp].best_node]
        triangles = node.mask.triangles.copy()
        print(f"size triangles before: {len(node.mask.triangles)}")
        for adj in node.adjacents:
            id_adj_comp = self.nodes[adj.id_node].mask.id_comp
            if id_adj_comp != id_comp: #this adjacent node is not in the same component
                #take the triangles in common
                overlap =  triangles & self.nodes[adj.id_node].mask.triangles
                if node.mask.ortho < self.nodes[adj.id_node].mask.ortho:
                    triangles = triangles - overlap 
        return len(triangles)/float(len( node.mask.triangles))
    
def load_mask(mask_path,name):
    global cameras
    mask_texture ,w,h= texture.load_texture(mask_path+"/"+name)

    image = Image.open(mask_path+"/"+name).convert('L')

    # Convert to a NumPy array (dtype will be uint8)
    img_data = np.array(image, dtype=np.uint32)

    img_data = erode_mask(img_data)

    m = re.match(r'(.+?)_(\d+)_(\d+)_([\d.]+)\.', name)
    
    img_name = m.group(1)  # Everything before the third last underscore
    
    for ic in range(0,len(cameras)) :
        if   (cameras[ic].label  == img_name):
            id_camera = ic  
            break


    X = int(m.group(2))    # Integer between third last and second last underscore
    Y = int(m.group(3))    # Integer between second last and last underscore
    C = float(m.group(4))  # Floating point number before the dot
    
    return mask(name,mask_texture, img_data, img_name,id_camera,w,h, X, Y, C)

def color_connected_component(id):
    component = all_masks.connected_components[id]
 
    for node_id in component:
        mask = all_masks.nodes[node_id].mask
        color_the_domain(mask, component.v_color)

def clear_domain():
    domain_mask.fill(0)
    tri_color.fill(0)

DBG_writeout = False
counter = 0
all_masks = mask_graph()
all_masks_GPU = mask_graph()

def compute_range(mks):
    global range_shader 
    global uv_map       # for each point in the img provides the coordinates of the point in parametric space
    global triangle_map  # for each point in the img provides the trinagle index
    global domain_mask  # keeps the coverage of the mask in parametric space
    global domain_mask_glob # it will replace domain_mask. IT keeps a pointer to a previous mask covering the same pixel
    global triangle_domain # it will replace domain_mask. IT keeps a pointer to a previous mask covering the same pixel
    global tri_color  #keeps the coverage of the mask in triangle space. tri_color[i*3,..,i*3+2] = [idt,idt,idt]
    global all_masks
    global counter
    global DBG_writeout
    global node_pointer
    global coverage
    global program_mask
    global triangle_map_texture
    global triangle_nodes
    global vertices
    global faces 
    global current_camera_matrix
    global texture_IMG_id
    global sensors
    
    # compute the max size of the masks
    max_mask_size = max(m.w * m.h for m in mks)

    #TEMP: number of processing masks with one dispatch 
    NMASKS = len(mks)
    n_wg = int(np.ceil(NMASKS/16))

    # create the node for the masks
    #all_masks.add_node(node(m))
    
    curr_node_id = len(all_masks.nodes)
    start_id = curr_node_id

    
    
    start_time = time.time()

    maskinfo_dtype = np.dtype([('index', np.int32, 4), ('corner', np.int32, 2),('_pad', np.int32, 2)])
    index_to_masks = np.zeros(NMASKS, dtype=maskinfo_dtype)

    for i, m in enumerate(mks):
        index_to_masks['index'][i] = (curr_node_id, (curr_node_id - start_id) * max_mask_size, m.w, m.h)
        index_to_masks['corner'][i] = (m.X, m.Y)
        curr_node_id += 1

    # ids = np.array([], dtype=np.int32)
    img_data = np.array([], dtype=np.uint32)
    for m in mks:
        img_data = np.append(img_data, m.img_data)
        img_data = np.append(img_data, np.zeros(max_mask_size - m.w * m.h, dtype=np.uint32))  # Fill the rest with zeros
        curr_node_id += 1
    del m
    elapsed_time = time.time() - start_time
    #print(f"Time spent in mask/ids array creation: {elapsed_time:.8f} seconds")


    #pass the index to the masks
    #ids = np.array([curr_node_id,0, m.w, m.h,m.X,m.Y,66,66], dtype=np.int32)
    start_time = time.time()

    indexToMasks_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexToMasks_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, index_to_masks.nbytes, index_to_masks, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, indexToMasks_ssbo)
  
    #pass the masks
    masks_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, masks_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, img_data.nbytes,  img_data, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, masks_ssbo)
     
   
    #average color
    rangeZ = np.zeros( n_wg*16, dtype=np.float32) # m.w*m.h to be replaced with the max value of all the masks
    rangeZ_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, rangeZ_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, rangeZ.nbytes, rangeZ, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, rangeZ_ssbo)
    
    glUseProgram(range_shader.program)

    sensor = sensors[cameras[mks[0].id_camera].sensor_id]
     
    glUniform1i(range_shader.uni("uMaskSize"), max_mask_size)
    glUniform1i(range_shader.uni("uNMasks"),NMASKS)
    glUniform1i(range_shader.uni("resolution_width"), sensor.resolution["width"])
    glUniform1i(range_shader.uni("resolution_height"), sensor.resolution["height"])

    glUniformMatrix4fv(range_shader.uni("uViewCam"), 1, GL_FALSE, glm.value_ptr(current_camera_matrix))

    elapsed_time = time.time() - start_time
    
    
    start_time = time.time()  
    glDispatchCompute(max(1,n_wg), 1 , 1)
    elapsed_time = time.time() - start_time
    print(f"Processed chunk of { (NMASKS)} masks in {elapsed_time:.8f} seconds")
    # Ensure compute shader completes
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    glFinish()  

    glUseProgram(0)


    #readback avgcol
    start_time = time.time()
    rangeZ = np.zeros(n_wg*16 , dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, rangeZ_ssbo)
    ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, rangeZ.nbytes, GL_MAP_READ_BIT)

    data_ptr = cast(ptr, POINTER(np.ctypeslib.ctypes.c_float))
    rangeZ[:] = np.ctypeslib.as_array(data_ptr, shape=(rangeZ.size,))
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    glDeleteBuffers(3, [indexToMasks_ssbo, masks_ssbo, rangeZ_ssbo])
    elapsed_time = time.time() - start_time
    #print(f"Time spent reading back rangeZ: {elapsed_time:.8f} seconds")
    

    # glBindTexture(GL_TEXTURE_2D, triangle_map_texture)
    # buf = np.empty((4000, 6000, 3), dtype=np.float32)
    # glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, buf)
    # uv_map_uint8 = (np.flipud(buf) * 0.0001).clip(0, 255).astype(np.uint8)
    # # save the color
    # image = Image.fromarray(uv_map_uint8 , 'RGB')
    # image.save(f"output_idtriangles_.png")    
    
    return rangeZ





def process_masks_GPU(mks,range_threshold = 10.0):
    global uv_map       # for each point in the img provides the coordinates of the point in parametric space
    global triangle_map  # for each point in the img provides the trinagle index
    global domain_mask  # keeps the coverage of the mask in parametric space
    global domain_mask_glob # it will replace domain_mask. IT keeps a pointer to a previous mask covering the same pixel
    global triangle_domain # it will replace domain_mask. IT keeps a pointer to a previous mask covering the same pixel
    global tri_color  #keeps the coverage of the mask in triangle space. tri_color[i*3,..,i*3+2] = [idt,idt,idt]
    global all_masks
    global counter
    global DBG_writeout
    global node_pointer
    global coverage
    global program_mask
    global triangle_map_texture
    global triangle_nodes
    global vertices
    global faces 
    global current_camera_matrix
    global texture_IMG_id
    global sensor 
    global sensor_FLUO 

    # compute the max size of the masks
    max_mask_size = max(m.w * m.h for m in mks)

    #TEMP: number of processing masks with one dispatch 
    NMASKS = len(mks)
    n_wg = int(np.ceil(NMASKS/16))
    # create the node for the masks
    #all_masks.add_node(node(m))
    
    curr_node_id = len(all_masks.nodes)
    start_id = curr_node_id

    maskinfo_dtype = np.dtype([('index', np.int32, 4), ('corner', np.int32, 2),('_pad', np.int32, 2)])
    index_to_masks = np.zeros(NMASKS, dtype=maskinfo_dtype)

    for i, m in enumerate(mks):
        index_to_masks['index'][i] = (curr_node_id, (curr_node_id - start_id) * max_mask_size, m.w, m.h)
        index_to_masks['corner'][i] = (m.X, m.Y)
        curr_node_id += 1


   
    img_data = np.array([], dtype=np.uint32)
    for m in mks:
        img_data = np.append(img_data, m.img_data)
        img_data = np.append(img_data, np.zeros(max_mask_size - m.w * m.h, dtype=np.uint32))  # Fill the rest with zeros
        curr_node_id += 1
    del m

    glActiveTexture(GL_TEXTURE12)
    glBindTexture(GL_TEXTURE_2D, texture_IMG_id)

    glActiveTexture(GL_TEXTURE13)
    glBindTexture(GL_TEXTURE_2D, triangle_map_texture)

    #pass the index to the masks
    #ids = np.array([curr_node_id,0, m.w, m.h,m.X,m.Y,66,66], dtype=np.int32)
    indexToMasks_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexToMasks_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, index_to_masks.nbytes, index_to_masks, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, indexToMasks_ssbo)
  
    #pass the masks
    masks_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, masks_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, img_data.nbytes,  img_data, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, masks_ssbo)
     
    #coverage ssbo
    cov = np.zeros( max_mask_size*n_wg*16, dtype=np.uint32) # m.w*m.h to be replaced with the max value of all the masks
    coverage_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coverage_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, cov.nbytes, cov, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, coverage_ssbo)
 
     #coverage ssbo weight
    cov_w = np.zeros( max_mask_size*n_wg*16, dtype=np.float32) # m.w*m.h to be replaced with the max value of all the masks
    coverage_weight_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coverage_weight_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, cov_w.nbytes, cov_w, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, coverage_weight_ssbo)

    #average color
    avg_col = np.zeros( 4*n_wg*16, dtype=np.float32) # m.w*m.h to be replaced with the max value of all the masks
    avg_col_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, avg_col_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, avg_col.nbytes, avg_col, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, avg_col_ssbo)
    
    glUseProgram(program_mask.program)

    glUniform1i(program_mask.uni("uMaskSize"), max_mask_size)

    sensor = sensors[cameras[mks[0].id_camera].sensor_id]

   # glUniform1i(program_mask.uni("uColorTexture"), 12)
    glUniformMatrix4fv(program_mask.uni("uViewCam"), 1, GL_FALSE, glm.value_ptr(current_camera_matrix))
    glUniform1f(program_mask.uni("uRangeThreshold"), range_threshold)
    glUniform1i(program_mask.uni("resolution_width"), sensor.resolution["width"])
    glUniform1i(program_mask.uni("resolution_height"), sensor.resolution["height"])

    #print("Current view matrix:\n", np.array(current_camera_matrix))
 
    #dbg just one to check the shader
    start_time = time.time()
    glDispatchCompute(max(1,n_wg), 1 , 1)
    
    # Ensure compute shader completes
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    glFinish()
    elapsed_time = time.time() - start_time
    
    global tot
    if 'tot' not in globals():
        tot = 0
    tot += NMASKS
    print(f"Processed chunk of {NMASKS} tot {tot}")

    glUseProgram(0)

    
    #readback coverage
    buf = np.zeros(max_mask_size*n_wg*16 , dtype=np.uint32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coverage_ssbo)
    ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buf.nbytes, GL_MAP_READ_BIT)

    data_ptr = cast(ptr, POINTER(c_uint32))
    buf[:] = np.ctypeslib.as_array(data_ptr, shape=(buf.size,))
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    #readback coverage weight
    buf_w = np.zeros(max_mask_size*n_wg*16  , dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, coverage_weight_ssbo)
    ptr_w = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buf_w.nbytes, GL_MAP_READ_BIT)

    data_ptr = cast(ptr_w, POINTER(np.ctypeslib.ctypes.c_float))
    buf_w[:] = np.ctypeslib.as_array(data_ptr, shape=(buf_w.size,))
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    #readback avgcol
    bufcol = np.zeros(4*n_wg*16 , dtype=np.float32)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, avg_col_ssbo)
    ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bufcol.nbytes, GL_MAP_READ_BIT)

    data_ptr = cast(ptr, POINTER(np.ctypeslib.ctypes.c_float))
    bufcol[:] = np.ctypeslib.as_array(data_ptr, shape=(bufcol.size,))
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    glDeleteBuffers(3, [indexToMasks_ssbo, masks_ssbo, coverage_ssbo])
     
    
    
     
    for i in range(NMASKS):
        num_elements = int(buf[i * max_mask_size])
        if num_elements == 0:
            continue

        mks[i].ones = num_elements
        mks[i].ones_w = float(buf_w[i * max_mask_size])*1000
        mks[i].avg_col = bufcol[i * 4:i * 4 + 3]

        mask_triangles = buf[i * max_mask_size + 1:i * max_mask_size + 1 + num_elements]
        weights =  buf_w[i * max_mask_size + 1:i * max_mask_size + 1 + num_elements]

        weighted_counter = Counter()
        for k, w in zip(mask_triangles, weights):
            weighted_counter[k] += int(w*1000)

        
        _ = Counter(mask_triangles)
        # mks[i].triangles = Counter(mask_triangles)
        mks[i].triangles = weighted_counter

        mks[i].n_triangles = len(mks[i].triangles)
    
        all_masks.add_node(node(mks[i]))
        curr_node_id = len(all_masks.nodes)-1

        adj_candidates = []
        for id_tri in mks[i].triangles.keys():
            for id_node in triangles_nodes[id_tri]:
                adj_candidates.append(id_node)
        adj_candidates = list(set(adj_candidates))

        t1 = all_masks.nodes[curr_node_id].mask.triangles
        for id_node in adj_candidates:
            t2 = all_masks.nodes[id_node].mask.triangles
            if all_masks.nodes[id_node].mask.ortho > all_masks.nodes[curr_node_id].mask.ortho:
                all_masks.nodes[curr_node_id].mask.triangles =  Counter({key: t1[key] for key in t1 if key not in t2})
                t1 = all_masks.nodes[curr_node_id].mask.triangles
            else:    
                all_masks.nodes[id_node].mask.triangles      =  Counter({key: t2[key] for key in t2 if key not in t1})



        for id_tri in  mks[i].triangles.keys():
            triangles_nodes[id_tri].append(curr_node_id)               

        mask_color = random_color()
        #color the triangles
        for id_tri in  mks[i].triangles.keys():
            c0 = id_tri*3*3
            tri_color[c0:c0+3] =   [color/255.0 for color in mask_color]
            tri_color[c0+3:c0+6] = [color/255.0 for color in mask_color]
            tri_color[c0+6:c0+9] = [color/255.0 for color in mask_color]

     
