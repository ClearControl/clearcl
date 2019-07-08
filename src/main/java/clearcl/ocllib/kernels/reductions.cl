__kernel
void reduce_min_buffer( DTYPE_IMAGE_IN_2D src,
                          long    length,
                          DTYPE_IMAGE_OUT_2D dst) 
{
  int index  = get_global_id(0);
  int stride = get_global_size(0);
  
  DTYPE_IN min = INFINITY;
  DTYPE_IN max = -INFINITY;
  
  while(index<length)
  {
    DTYPE_IN value = src[index];
    min = fmin(min, value);
    max = fmax(max, value);
    index += stride;
  }

  dst[2*get_global_id(0)+0] = min;
  dst[2*get_global_id(0)+1] = max;
}

