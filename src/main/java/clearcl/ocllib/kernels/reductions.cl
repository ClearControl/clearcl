__kernel void reduce_minmax_1d( DTYPE_IMAGE_IN_2D src, DTYPE_IMAGE_OUT_2D dst,
                          long  length)
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

__kernel void reduce_minmax_2d( DTYPE_IMAGE_IN_2D src, DTYPE_IMAGE_OUT_2D dst,
                          long  length)
{
  const int width = GET_IMAGE_WIDTH(src);
  const int height = GET_IMAGE_HEIGHT(src);

  const int x = get_global_id(0);
  const int y = get_global_id(1);  
  const int stridex = get_global_size(0);
  const int stridey = get_global_size(1);
  
  DTYPE_IN min = INFINITY;
  DTYPE_IN max = -INFINITY;
  
  for(int ly=y; ly<height; ly+=stridey)
  {
    for(int lx=x; lx<width; lx+=stridex)
    {
      const int2 pos = {lx,ly};
      const DTYPE_OUT value = READ_IMAGE_2D(src, sampler, pos);
  
      min = fmin(min, value);
      max = fmax(max, value);
    }
  }

  dst[2*get_global_id(0)+0] = min;
  dst[2*get_global_id(0)+1] = max;
}

