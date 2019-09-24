package clearcl.ops.kernels;

import java.util.HashMap;
import java.util.Map;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import clearcl.enums.ImageChannelDataType;
import coremem.enums.NativeTypeEnum;

/**
 * TypeFixer
 *
 * Author: @haesleinhuepf 06 2019
 */
public class TypeFixer
{
  CLKernelExecutor clke;

  Map<String, Object> map;

  Map<String, Object> inputMap = new HashMap<>();
  Map<String, Object> outputMap = new HashMap<>();

  public TypeFixer(CLKernelExecutor clij, Map<String, Object> map)
  {
    this.map = map;
    this.clke = clij;

    if (map.size() < 3)
    {
      return;
    }

    for (String key : map.keySet())
    {
      Object object = map.get(key);
      if (object instanceof ClearCLBuffer
          || object instanceof ClearCLImage)
      {
        if ((key.contains("src") || key.contains("input")))
        {
          inputMap.put(key, object);
        }
        else
        {
          outputMap.put(key, object);
        }
      }
    }
  }

  public void fix() throws CLKernelException
  {
    fix(inputMap);
    fix(outputMap);
  }

  public void unfix() throws CLKernelException
  {
    unfix(inputMap);
    unfix(outputMap);
  }

  private void fix(Map<String, Object> currentMap) throws CLKernelException
  {
    if (currentMap.size() < 2)
    {
      currentMap.clear();
      return;
    }

    boolean fixNecessary = false;
    NativeTypeEnum type = null;
    for (String key : currentMap.keySet())
    {
      Object object = currentMap.get(key);
      NativeTypeEnum currentType = null;
      if (object instanceof ClearCLImage)
      {
        currentType = ((ClearCLImage) object).getNativeType();
      }
      else if (object instanceof ClearCLBuffer)
      {
        currentType = ((ClearCLBuffer) object).getNativeType();
      }
      if (type == null)
      {
        type = currentType;
      }
      else
      {
        if (type != currentType)
        {
          fixNecessary = true;
        }
      }
    }

    if (!fixNecessary)
    {
      currentMap.clear();
      return;
    }

    for (String key : currentMap.keySet())
    {
      Object object = currentMap.get(key);
      if (object instanceof ClearCLImage)
      {
        ClearCLImage inImage = (ClearCLImage) object;
        if (inImage.getNativeType() != NativeTypeEnum.Float)
        {
          ClearCLImage image =
                             clke.createCLImage(inImage.getDimensions(),
                                                ImageChannelDataType.Float);
          if (currentMap == inputMap)
          {
            Kernels.copy(clke, inImage, image);
          }
          map.remove(key);
          map.put(key, image);
        }

      }
      else if (object instanceof ClearCLBuffer)
      {

        ClearCLBuffer inBuffer = (ClearCLBuffer) object;
        if (inBuffer.getNativeType() != NativeTypeEnum.Float)
        {

          ClearCLBuffer buffer =
                               clke.createCLBuffer(inBuffer.getDimensions(),
                                                   NativeTypeEnum.Float);
          if (currentMap == inputMap)
          {
            Kernels.copy(clke, inBuffer, buffer);
          }
          map.remove(key);
          map.put(key, buffer);
        }
      }
    }
  }

  public void unfix(Map<String, Object> currentMap) throws CLKernelException
  {
    for (String key : currentMap.keySet())
    {
      Object object = currentMap.get(key);
      Object origin = map.get(key);
      if (object != origin)
      {
        if (object instanceof ClearCLImage)
        {
          ClearCLImage outImage = (ClearCLImage) object;
          ClearCLImage image = (ClearCLImage) origin;
          if (currentMap == outputMap)
          {
            Kernels.copy(clke, image, outImage);
          }
          image.close();
          map.remove(key);
          map.put(key, outImage);
        }
        else if (object instanceof ClearCLBuffer)
        {
          ClearCLBuffer outBuffer = (ClearCLBuffer) object;
          ClearCLBuffer buffer = (ClearCLBuffer) origin;
          if (currentMap == outputMap)
          {
            Kernels.copy(clke, buffer, outBuffer);
          }
          buffer.close();
          map.remove(key);
          map.put(key, outBuffer);
        }
      }
    }
    currentMap.clear();
  }
}
