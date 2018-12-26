#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

#pragma OPENCL EXTENSION cl_amd_printf : enable

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#ifndef M_PI
    #define   M_PI 3.14159265358979323846f /* pi */
#endif

#ifndef M_LOG2E
    #define   M_LOG2E   1.4426950408889634074f /* log_2 e */
#endif
 
#ifndef M_LOG10E
    #define   M_LOG10E   0.43429448190325182765f /* log_10 e */
#endif
 
#ifndef M_LN2
    #define   M_LN2   0.69314718055994530942f  /* log_e 2 */
#endif

#ifndef M_LN10
    #define   M_LN10   2.30258509299404568402f /* log_e 10 */
#endif

#ifndef BUFFER_READ_WRITE
    #define BUFFER_READ_WRITE 1

inline char2 read_buffer3dc(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global char * buffer_var, sampler_t sampler, int4 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (char2){0, 0};
    }
    return (char2){buffer_var[pos_in_buffer],0};
}

inline uchar2 read_buffer3duc(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int4 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (uchar2){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}

inline short2 read_buffer3di(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global short * buffer_var, sampler_t sampler, int4 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (short2){0, 0};
    }
    return (short2){buffer_var[pos_in_buffer],0};
}

inline ushort2 read_buffer3dui(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global ushort * buffer_var, sampler_t sampler, int4 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (ushort2){0, 0};
    }
    return (ushort2){buffer_var[pos_in_buffer],0};
}

inline float2 read_buffer3df(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global float* buffer_var, sampler_t sampler, int4 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width + pos.z * read_buffer_width * read_buffer_height;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height || pos.z < 0 || pos.z >= read_buffer_depth) {
        return (float2){0, 0};
    }
    return (float2){buffer_var[pos_in_buffer],0};
}

inline void write_buffer3dc(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global char * buffer_var, int4 pos, char value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer3duc(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global uchar * buffer_var, int4 pos, uchar value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer3di(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global short * buffer_var, int4 pos, short value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer3dui(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global ushort * buffer_var, int4 pos, ushort value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer3df(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global float* buffer_var, int4 pos, float value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width + pos.z * write_buffer_width * write_buffer_height;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height || pos.z < 0 || pos.z >= write_buffer_depth) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline char2 read_buffer2dc(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global char * buffer_var, sampler_t sampler, int2 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (char){0, 0};
    }
    return (char2){buffer_var[pos_in_buffer],0};
}

inline uchar2 read_buffer2duc(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global uchar * buffer_var, sampler_t sampler, int2 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (uchar){0, 0};
    }
    return (uchar2){buffer_var[pos_in_buffer],0};
}

inline short2 read_buffer2di(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global short * buffer_var, sampler_t sampler, int2 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (short2){0, 0};
    }
    return (short2){buffer_var[pos_in_buffer],0};
}

inline ushort2 read_buffer2dui(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global ushort * buffer_var, sampler_t sampler, int2 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (ushort2){0, 0};
    }
    return (ushort2){buffer_var[pos_in_buffer],0};
}

inline float2 read_buffer2df(int read_buffer_width, int read_buffer_height, int read_buffer_depth, __global float* buffer_var, sampler_t sampler, int2 pos )
{
    int pos_in_buffer = pos.x + pos.y * read_buffer_width;
    if (pos.x < 0 || pos.x >= read_buffer_width || pos.y < 0 || pos.y >= read_buffer_height) {
        return (float2){0, 0};
    }
    return (float2){buffer_var[pos_in_buffer],0};
}

inline void write_buffer2dc(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global char * buffer_var, int2 pos, char value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer2duc(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global uchar * buffer_var, int2 pos, uchar value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer2di(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global short * buffer_var, int2 pos, short value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer2dui(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global ushort * buffer_var, int2 pos, ushort value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline void write_buffer2df(int write_buffer_width, int write_buffer_height, int write_buffer_depth, __global float* buffer_var, int2 pos, float value )
{
    int pos_in_buffer = pos.x + pos.y * write_buffer_width;
    if (pos.x < 0 || pos.x >= write_buffer_width || pos.y < 0 || pos.y >= write_buffer_height) {
        return;
    }
    buffer_var[pos_in_buffer] = value;
}

inline int get_bufferc_width(int size, __global char* buffer_var )
{
    return size;
}
inline int get_bufferuc_width(int size, __global uchar* buffer_var )
{
    return size;
}
inline int get_bufferi_width(int size, __global short* buffer_var )
{
    return size;
}
inline int get_bufferui_width(int size, __global ushort* buffer_var )
{
    return size;
}
inline int get_bufferf_width(int size, __global float* buffer_var )
{
    return size;
}

inline int get_bufferuc_height(int size, __global char* buffer_var )
{
    return size;
}
inline int get_bufferuuc_height(int size, __global uchar* buffer_var )
{
    return size;
}
inline int get_bufferi_height(int size, __global short* buffer_var )
{
    return size;
}
inline int get_bufferui_height(int size, __global ushort* buffer_var )
{
    return size;
}
inline int get_bufferf_height(int size, __global float* buffer_var )
{
    return size;
}

inline int get_bufferc_depth(int size, __global char* buffer_var )
{
    return size;
}
inline int get_bufferuc_depth(int size, __global uchar* buffer_var )
{
    return size;
}
inline int get_bufferi_depth(int size, __global short* buffer_var )
{
    return size;
}
inline int get_bufferui_depth(int size, __global ushort* buffer_var )
{
    return size;
}
inline int get_bufferf_depth(int size, __global float* buffer_var )
{
    return size;
}
#endif