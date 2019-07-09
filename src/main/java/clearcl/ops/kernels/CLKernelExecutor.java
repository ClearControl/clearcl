package clearcl.ops.kernels;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLContext;
import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.ClearCLProgram;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.ImageChannelOrder;
import clearcl.enums.KernelAccessType;
import clearcl.enums.MemAllocMode;
import clearcl.exceptions.OpenCLException;
import coremem.enums.NativeTypeEnum;

/**
 * This executor can call OpenCL files. It uses some functionality adapted from
 * FastFuse, to make .cl file handling easier. For example, it ensures that the
 * right image_read/image_write methods are called depending on the image type.
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG
 * (http://mpi-cbg.de) February 2018
 */
public class CLKernelExecutor
{
  private static final boolean DEBUG = false;
  public static int MAX_ARRAY_SIZE = 1000;

  private final ClearCLContext context;
  private Class anchorClass;

  private final HashMap<String, ClearCLProgram> programCacheMap =
                                                                new HashMap<>();
  private final HashMap<String, ArrayList<String>> variableListMap =
                                                                   new HashMap<>();
  private final HashMap<String, String> sourceCodeCache =
                                                        new HashMap<>();

  public CLKernelExecutor(ClearCLContext context,
                          Class anchorClass) throws IOException
  {
    super();
    this.context = context;
    this.anchorClass = anchorClass;
  }

  public static void getOpenCLDefines(Map<String, Object> defines,
                                      ImageChannelDataType imageChannelDataType,
                                      boolean isInputImage)
  {
    if (isInputImage)
    {
      defines.put("DTYPE_IMAGE_IN_3D", "__read_only image3d_t");
      defines.put("DTYPE_IMAGE_IN_2D", "__read_only image2d_t");
      if (imageChannelDataType.isInteger())
      {
        if (imageChannelDataType == ImageChannelDataType.UnsignedInt8
            || imageChannelDataType == ImageChannelDataType.SignedInt8)
        {
          defines.put("DTYPE_IN", "char");
        }
        else
        {
          defines.put("DTYPE_IN", "ushort");
        }
      }
      else
      {
        defines.put("DTYPE_IN", "float");
      }
      defines.put("READ_IMAGE_2D",
                  imageChannelDataType.isInteger() ? "read_imageui"
                                                   : "read_imagef");
      defines.put("READ_IMAGE_3D",
                  imageChannelDataType.isInteger() ? "read_imageui"
                                                   : "read_imagef");
    }
    else
    {
      defines.put("DTYPE_IMAGE_OUT_3D", "__write_only image3d_t");
      defines.put("DTYPE_IMAGE_OUT_2D", "__write_only image2d_t");
      if (imageChannelDataType.isInteger())
      {
        if (imageChannelDataType == ImageChannelDataType.UnsignedInt8
            || imageChannelDataType == ImageChannelDataType.SignedInt8)
        {
          defines.put("DTYPE_OUT", "char");
        }
        else
        {
          defines.put("DTYPE_OUT", "ushort");
        }
      }
      else
      {
        defines.put("DTYPE_OUT", "float");
      }
      defines.put("WRITE_IMAGE_2D",
                  imageChannelDataType.isInteger() ? "write_imageui"
                                                   : "write_imagef");
      defines.put("WRITE_IMAGE_3D",
                  imageChannelDataType.isInteger() ? "write_imageui"
                                                   : "write_imagef");
    }
  }

  public static void getOpenCLDefines(Map<String, Object> defines,
                                      NativeTypeEnum nativeTypeEnum,
                                      boolean isInputImage)
  {
    String typeName = nativeTypeToOpenCLTypeName(nativeTypeEnum);
    String typeId = nativeTypeToOpenCLTypeShortName(nativeTypeEnum);

    if (isInputImage)
    {
      defines.put("DTYPE_IN", typeName);
      defines.put("DTYPE_IMAGE_IN_3D", "__global " + typeName + "*");
      defines.put("DTYPE_IMAGE_IN_2D", "__global " + typeName + "*");
      defines.put("READ_IMAGE_2D(a,b,c)", "read_buffer2d" + typeId
                                          + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)");
      defines.put("READ_IMAGE_3D(a,b,c)", "read_buffer3d" + typeId
                                          + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)");
    }
    else
    {
      defines.put("DTYPE_OUT", typeName);
      defines.put("DTYPE_IMAGE_OUT_3D", "__global " + typeName + "*");
      defines.put("DTYPE_IMAGE_OUT_2D", "__global " + typeName + "*");
      defines.put("WRITE_IMAGE_2D(a,b,c)", "write_buffer2d" + typeId
                                           + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)");
      defines.put("WRITE_IMAGE_3D(a,b,c)", "write_buffer3d" + typeId
                                           + "(GET_IMAGE_WIDTH(a),GET_IMAGE_HEIGHT(a),GET_IMAGE_DEPTH(a),a,b,c)");
    }
  }

  private static String nativeTypeToOpenCLTypeName(NativeTypeEnum pDType)
  {
    if (null == pDType)
    {
      return "";
    }
    else
    {
      switch (pDType)
      {
      case Byte:
        return "char";
      case UnsignedByte:
        return "uchar";
      case Short:
        return "short";
      case UnsignedShort:
        return "ushort";
      case Float:
        return "float";
      default:
        return "";
      }
    }
  }

  private static String nativeTypeToOpenCLTypeShortName(NativeTypeEnum pDType)
  {
    if (null == pDType)
    {
      return "";
    }
    else
    {
      switch (pDType)
      {
      case Byte:
        return "c";
      case UnsignedByte:
        return "uc";
      case Short:
        return "i";
      case UnsignedShort:
        return "ui";
      case Float:
        return "f";
      default:
        return "";
      }
    }
  }

  public ClearCLBuffer createCLBuffer(ClearCLBuffer inputCL)
  {
    return createCLBuffer(inputCL.getDimensions(),
                          inputCL.getNativeType());
  }

  public ClearCLBuffer createCLBuffer(long[] dimensions,
                                      NativeTypeEnum pNativeType)
  {
    return context.createBuffer(MemAllocMode.Best,
                                HostAccessType.ReadWrite,
                                KernelAccessType.ReadWrite,
                                1L,
                                pNativeType,
                                dimensions);
  }

  public ClearCLImage createCLImage(ClearCLImage pInputImage)
  {
    return context.createImage(pInputImage);
  }

  public ClearCLImage createCLImage(long[] dimensions,
                                    ImageChannelDataType pImageChannelType)
  {
    return context.createImage(HostAccessType.ReadWrite,
                               KernelAccessType.ReadWrite,
                               ImageChannelOrder.R,
                               pImageChannelType,
                               dimensions);
  }

  public void execute(String pProgramFilename,
                      String pKernelname,
                      Map<String, Object> pParameterMap) throws CLKernelException
  {
    execute(Object.class,
            pProgramFilename,
            pKernelname,
            pParameterMap);
  }

  public void execute(Class pAnchorClass,
                      String pProgramFilename,
                      String pKernelname,
                      Map<String, Object> pParameterMap) throws CLKernelException
  {
    execute(pAnchorClass,
            pProgramFilename,
            pKernelname,
            null,
            pParameterMap);
  }

  public void execute(Class pAnchorClass,
                      String pProgramFilename,
                      String pKernelname,
                      long[] pGlobalsizes,
                      Map<String, Object> pParameterMap) throws CLKernelException
  {
    // TODO: How efficient is it to construct a new instance everytime?
    // Use a single instance, and set the parameterMap? That would need
    // locking..
    // Use static function?
    TypeFixer inputTypeFixer = new TypeFixer(this, pParameterMap);
    // TODO: this function solely relies on side effects. Give it input and
    // output
    inputTypeFixer.fix();

    if (DEBUG)
    {
      for (String key : pParameterMap.keySet())
      {
        System.out.println(key + " = " + pParameterMap.get(key));
      }
    }

    this.setAnchorClass(pAnchorClass);
    this.enqueue(true,
                 pProgramFilename,
                 pKernelname,
                 pParameterMap,
                 pGlobalsizes);
  }

  /**
   * Function that does the actual work
   * 
   * @param waitToFinish
   *          Synchronous or asynchronous execution
   * @param programFilename
   *          Name of the file containing the kernel code
   * 
   * @param kernelName
   *          Name of the kernel inside the programFile
   * @param parameterMap
   *          Map of all parameters. It is recommended that input and output
   *          images are given with the names "src" and "dst", respectively.
   * @param globalSizes
   *          Dimensions of the input image
   * @throws CLKernelException
   */
  public void enqueue(boolean waitToFinish,
                      String programFilename,
                      String kernelName,
                      Map<String, Object> parameterMap,
                      long[] globalSizes) throws CLKernelException
  {
    if (DEBUG)
    {
      System.out.println("Loading " + kernelName);
    }

    ClearCLImage srcImage = null;
    ClearCLImage dstImage = null;
    ClearCLBuffer srcBuffer = null;
    ClearCLBuffer dstBuffer = null;

    if (parameterMap != null)
    {
      for (String key : parameterMap.keySet())
      {
        if (parameterMap.get(key) instanceof ClearCLImage)
        {
          if (key.contains("src") || key.contains("input"))
          {
            srcImage = (ClearCLImage) parameterMap.get(key);
          }
          else if (key.contains("dst") || key.contains("output"))
          {
            dstImage = (ClearCLImage) parameterMap.get(key);
          }
        }
        else if (parameterMap.get(key) instanceof ClearCLBuffer)
        {
          if (key.contains("src") || key.contains("input"))
          {
            srcBuffer = (ClearCLBuffer) parameterMap.get(key);
          }
          else if (key.contains("dst") || key.contains("output"))
          {
            dstBuffer = (ClearCLBuffer) parameterMap.get(key);
          }
        }
      }
    }

    if (dstImage == null && dstBuffer == null)
    {
      if (srcImage != null)
      {
        dstImage = srcImage;
      }
      else if (srcBuffer != null)
      {
        dstBuffer = srcBuffer;
      }
    }
    else if (srcImage == null && srcBuffer == null)
    {
      if (dstImage != null)
      {
        srcImage = dstImage;
      }
      else if (dstBuffer != null)
      {
        srcBuffer = dstBuffer;
      }
    }

    Map<String, Object> openCLDefines = new HashMap<>();
    openCLDefines.put("MAX_ARRAY_SIZE", MAX_ARRAY_SIZE); // needed for median.
                                                         // Median is limited to
                                                         // a given array length
                                                         // to be sorted
    if (srcImage != null)
    {
      getOpenCLDefines(openCLDefines,
                       srcImage.getChannelDataType(),
                       true);
    }
    if (dstImage != null)
    {
      getOpenCLDefines(openCLDefines,
                       dstImage.getChannelDataType(),
                       false);
    }
    if (srcBuffer != null)
    {
      getOpenCLDefines(openCLDefines,
                       srcBuffer.getNativeType(),
                       true);
    }
    if (dstBuffer != null)
    {
      getOpenCLDefines(openCLDefines,
                       dstBuffer.getNativeType(),
                       false);
    }

    // deal with image width/height/depth for all images and buffers
    ArrayList<String> definedParameterKeys = new ArrayList<>();
    for (String key : parameterMap.keySet())
    {
      if (parameterMap.get(key) instanceof ClearCLImage)
      {
        ClearCLImage image = (ClearCLImage) parameterMap.get(key);
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_WIDTH",
                          image.getWidth());
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_HEIGHT",
                          image.getHeight());
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_DEPTH",
                          image.getDepth());
      }
      else if (parameterMap.get(key) instanceof ClearCLBuffer)
      {
        ClearCLBuffer image = (ClearCLBuffer) parameterMap.get(key);
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_WIDTH",
                          image.getWidth());
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_HEIGHT",
                          image.getHeight());
        openCLDefines.put("IMAGE_SIZE_" + key
                          + "_DEPTH",
                          image.getDepth());
      }
      definedParameterKeys.add(key);
    }

    openCLDefines.put("GET_IMAGE_IN_WIDTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _WIDTH");
    openCLDefines.put("GET_IMAGE_IN_HEIGHT(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _HEIGHT");
    openCLDefines.put("GET_IMAGE_IN_DEPTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _DEPTH");
    openCLDefines.put("GET_IMAGE_OUT_WIDTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _WIDTH");
    openCLDefines.put("GET_IMAGE_OUT_HEIGHT(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _HEIGHT");
    openCLDefines.put("GET_IMAGE_OUT_DEPTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _DEPTH");
    openCLDefines.put("GET_IMAGE_WIDTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _WIDTH");
    openCLDefines.put("GET_IMAGE_HEIGHT(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _HEIGHT");
    openCLDefines.put("GET_IMAGE_DEPTH(image_key)",
                      "IMAGE_SIZE_ ## image_key ## _DEPTH");

    // add undefined parameters to define list
    ArrayList<String> variableNames =
                                    getImageVariablesFromSource(programFilename);
    for (String variableName : variableNames)
    {

      boolean existsAlready = false;
      for (String key : definedParameterKeys)
      {
        if (key.compareTo(variableName) == 0)
        {
          existsAlready = true;
          break;
        }
      }
      if (!existsAlready)
      {
        openCLDefines.put("IMAGE_SIZE_" + variableName + "_WIDTH", 0);
        openCLDefines.put("IMAGE_SIZE_" + variableName
                          + "_HEIGHT",
                          0);
        openCLDefines.put("IMAGE_SIZE_" + variableName + "_DEPTH", 0);
      }
    }

    if (DEBUG)
    {
      for (String key : openCLDefines.keySet())
      {
        System.out.println(key + " = " + openCLDefines.get(key));
      }
    }

    ClearCLKernel clearCLKernel;

    try
    {
      clearCLKernel = getKernel(context,
                                programFilename,
                                kernelName,
                                openCLDefines);
    }
    catch (IOException e1)
    {
      throw new CLKernelException("IOException accessing clearCLKernal: "
                                  + kernelName);
    }

    if (clearCLKernel != null)
    {
      if (globalSizes != null)
      {
        clearCLKernel.setGlobalSizes(globalSizes);
      }
      else if (dstImage != null)
      {
        clearCLKernel.setGlobalSizes(dstImage.getDimensions());
      }
      else if (dstBuffer != null)
      {
        clearCLKernel.setGlobalSizes(dstBuffer.getDimensions());
      }
      if (parameterMap != null)
      {
        for (String key : parameterMap.keySet())
        {
          clearCLKernel.setArgument(key, parameterMap.get(key));
        }
      }
      if (DEBUG)
      {
        System.out.println("Executing " + kernelName);
      }

      try
      {
        clearCLKernel.run(waitToFinish);
      }
      catch (Exception e)
      {
        throw new CLKernelException("",
                                    (clearCLKernel.getSourceCode()));
      }

      /* 
       // If it is desired to measure execution times...
      // Would need to rewrite the measure function to throw exceptions
      final ClearCLKernel kernel = clearCLKernel;
      double duration = ElapsedTime.measure("Pure kernel execution", new Runnable() {
        @Override
        public void run() {
          try
          {
            kernel.run(waitToFinish);
          }
          catch (Exception e)
          {
            throw new CLKernelException ("", (kernel.getSourceCode()));
          }
        }
      });
      if (DEBUG)
      {
        System.out.println("Returned from " + kernelName
                           + " after "
                           + duration
                           + " msec");
      }
      */
      clearCLKernel.close();
    }

  }

  private ArrayList<String> getImageVariablesFromSource(String programFilename)
  {
    String key = anchorClass.getName() + "_" + programFilename;

    if (variableListMap.containsKey(key))
    {
      return variableListMap.get(key);
    }
    ArrayList<String> variableList = new ArrayList<>();

    String sourceCode = getProgramSource(programFilename);
    String[] kernels = sourceCode.split("__kernel");

    kernels[0] = "";
    for (String kernel : kernels)
    {
      if (kernel.length() > 0)
      {
        String temp1 = kernel.split("\\(")[1];
        if (temp1.length() > 0)
        {
          String parameterText = temp1.split("\\)")[0];
          parameterText = parameterText.replace("\n", " ");
          parameterText = parameterText.replace("\t", " ");
          parameterText = parameterText.replace("\r", " ");

          String[] parameters = parameterText.split(",");
          for (String parameter : parameters)
          {
            if (parameter.contains("IMAGE"))
            {
              String[] temp2 = parameter.trim().split(" ");
              String variableName = temp2[temp2.length - 1];

              variableList.add(variableName);

            }
          }
        }
      }
    }

    variableListMap.put(key, variableList);
    return variableList;
  }

  protected String getProgramSource(String programFilename)
  {
    String key = anchorClass.getName() + "_" + programFilename;

    if (sourceCodeCache.containsKey(key))
    {
      return sourceCodeCache.get(key);
    }
    try
    {
      ClearCLProgram program = context.createProgram(this.anchorClass,
                                                     new String[]
                                                     { programFilename });
      String source = program.getSourceCode();
      sourceCodeCache.put(key, source);
      return source;
    }
    catch (IOException e)
    {
      // e.printStackTrace();
      System.out.println("IOException creating program: "
                         + programFilename);
    }
    return "";
  }

  public void setAnchorClass(Class anchorClass)
  {
    this.anchorClass = anchorClass;
  }

  protected ClearCLKernel getKernel(ClearCLContext context,
                                    String programFilename,
                                    String kernelName,
                                    Map<String, Object> defines) throws IOException,
                                                                 NullPointerException,
                                                                 CLKernelException
  {
    String programCacheKey = anchorClass.getCanonicalName() + " "
                             + programFilename;
    for (String key : defines.keySet())
    {
      programCacheKey = programCacheKey + " "
                        + (key + " = " + defines.get(key));
    }
    if (DEBUG)
    {
      System.out.println("Program cache hash:" + programCacheKey);
    }
    ClearCLProgram clProgram =
                             this.programCacheMap.get(programCacheKey);
    if (clProgram == null)
    {
      clProgram = context.createProgram(this.anchorClass, new String[]
      { programFilename });
      for (Map.Entry<String, Object> entry : defines.entrySet())
      {
        if (entry.getValue() instanceof String)
        {
          clProgram.addDefine((String) entry.getKey(),
                              (String) entry.getValue());
        }
        else if (entry.getValue() instanceof Number)
        {
          clProgram.addDefine((String) entry.getKey(),
                              (Number) entry.getValue());
        }
        else if (entry.getValue() == null)
        {
          clProgram.addDefine((String) entry.getKey());
        }
      }

      clProgram.addBuildOptionAllMathOpt();
      clProgram.buildAndLog();

      programCacheMap.put(programCacheKey, clProgram);
    }

    try
    {
      return clProgram.createKernel(kernelName);
    }
    catch (OpenCLException e)
    {
      throw new CLKernelException("Error creating kernel: " + kernelName);
    }
  }

  public void close()
  {
    for (String key : programCacheMap.keySet())
    {
      ClearCLProgram program = programCacheMap.get(key);
      program.close();
    }
    programCacheMap.clear();
  }
}
