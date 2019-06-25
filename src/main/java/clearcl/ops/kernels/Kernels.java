package clearcl.ops.kernels;

import static clearcl.ops.kernels.KernelUtils.radiusToKernelSize;
import static clearcl.ops.kernels.KernelUtils.sigmaToKernelSize;

import java.nio.FloatBuffer;
import java.util.HashMap;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLImage;
import clearcl.ocllib.kernels.CLKernels;
import coremem.enums.NativeTypeEnum;

/**
 * This class contains convenience access functions for OpenCL based image
 * processing.
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG
 * (http://mpi-cbg.de) March 2018
 */
public class Kernels
{

  public static boolean absolute(CLKernelExecutor clke,
                                 ClearCLImage src,
                                 ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "absolute_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean absolute(CLKernelExecutor clke,
                                 ClearCLBuffer src,
                                 ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "absolute_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean addImages(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage src1,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean addImages(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer src1,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean addImageAndScalar(CLKernelExecutor clke,
                                          ClearCLImage src,
                                          ClearCLImage dst,
                                          Float scalar)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addScalar_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean addImageAndScalar(CLKernelExecutor clke,
                                          ClearCLBuffer src,
                                          ClearCLBuffer dst,
                                          Float scalar)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addScalar_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean addImagesWeighted(CLKernelExecutor clke,
                                          ClearCLImage src,
                                          ClearCLImage src1,
                                          ClearCLImage dst,
                                          Float factor,
                                          Float factor1)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("factor", factor);
    parameters.put("factor1", factor1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addWeightedPixelwise_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean addImagesWeighted(CLKernelExecutor clke,
                                          ClearCLBuffer src,
                                          ClearCLBuffer src1,
                                          ClearCLBuffer dst,
                                          Float factor,
                                          Float factor1)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("factor", factor);
    parameters.put("factor1", factor1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "addWeightedPixelwise_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean affineTransform(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst,
                                        float[] matrix)
  {

    ClearCLBuffer matrixCl = clke.createCLBuffer(new long[]
    { matrix.length, 1, 1 }, NativeTypeEnum.Float);

    FloatBuffer buffer = FloatBuffer.wrap(matrix);
    matrixCl.readFrom(buffer, true);

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("input", src);
    parameters.put("output", dst);
    parameters.put("mat", matrixCl);

    boolean result = clke.execute(CLKernels.class,
                                  "affineTransforms.cl",
                                  "affine",
                                  parameters);

    matrixCl.close();

    return result;
  }
  /*
    public static boolean affineTransform(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, AffineTransform3D at) {
        at = at.inverse();
        float[] matrix = AffineTransform.matrixToFloatArray(at);
        return affineTransform(clke, src, dst, matrix);
    }
    */

  public static boolean affineTransform(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst,
                                        float[] matrix)
  {

    ClearCLBuffer matrixCl = clke.createCLBuffer(new long[]
    { matrix.length, 1, 1 }, NativeTypeEnum.Float);

    FloatBuffer buffer = FloatBuffer.wrap(matrix);
    matrixCl.readFrom(buffer, true);

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("input", src);
    parameters.put("output", dst);
    parameters.put("mat", matrixCl);

    boolean result = clke.execute(CLKernels.class,
                                  "affineTransforms_interpolate.cl",
                                  "affine_interpolate",
                                  parameters);

    matrixCl.close();

    return result;
  }

  /*
  public static boolean affineTransform(CLKernelExecutor clke, ClearCLImage src, ClearCLImage dst, AffineTransform3D at) {
      at = at.inverse();
      float[] matrix = AffineTransform.matrixToFloatArray(at);
      return affineTransform(clke, src, dst, matrix);
  }
  */

  public static boolean applyVectorfield(CLKernelExecutor clke,
                                         ClearCLImage src,
                                         ClearCLImage vectorX,
                                         ClearCLImage vectorY,
                                         ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);

    boolean result =
                   clke.execute(CLKernels.class,
                                "deform_interpolate.cl",
                                "deform_2d_interpolate",
                                parameters);
    return result;
  }

  public static boolean applyVectorfield(CLKernelExecutor clke,
                                         ClearCLImage src,
                                         ClearCLImage vectorX,
                                         ClearCLImage vectorY,
                                         ClearCLImage vectorZ,
                                         ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);
    parameters.put("vectorZ", vectorZ);

    boolean result =
                   clke.execute(CLKernels.class,
                                "deform_interpolate.cl",
                                "deform_3d_interpolate",
                                parameters);
    return result;
  }

  public static boolean applyVectorfield(CLKernelExecutor clke,
                                         ClearCLBuffer src,
                                         ClearCLBuffer vectorX,
                                         ClearCLBuffer vectorY,
                                         ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);

    boolean result =
                   clke.execute(CLKernels.class,
                                "deform.cl",
                                "deform_2d",
                                parameters);
    return result;
  }

  public static boolean applyVectorfield(CLKernelExecutor clke,
                                         ClearCLBuffer src,
                                         ClearCLBuffer vectorX,
                                         ClearCLBuffer vectorY,
                                         ClearCLBuffer vectorZ,
                                         ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);
    parameters.put("vectorZ", vectorZ);

    boolean result =
                   clke.execute(CLKernels.class,
                                "deform.cl",
                                "deform_3d",
                                parameters);
    return result;
  }

  /*
  public static boolean automaticThreshold(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, String userSelectedMethod) {
      Float minimumGreyValue = 0f;
      Float maximumGreyValue = 0f;
      Integer numberOfBins = 256;
  
      if (src.getNativeType() == NativeTypeEnum.UnsignedByte) {
          minimumGreyValue = 0f;
          maximumGreyValue = 255f;
      } else {
          minimumGreyValue = null;
          maximumGreyValue = null;
      }
  
      return automaticThreshold(clke, src, dst, userSelectedMethod, minimumGreyValue, maximumGreyValue, 256);
  }
  
  
  public static boolean automaticThreshold(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, String userSelectedMethod, Float minimumGreyValue, Float maximumGreyValue, Integer numberOfBins) {
  
      if (minimumGreyValue == null)
      {
          minimumGreyValue = new Double(Kernels.minimumOfAllPixels(clke, src)).floatValue();
      }
  
      if (maximumGreyValue == null)
      {
          maximumGreyValue = new Double(Kernels.maximumOfAllPixels(clke, src)).floatValue();
      }
  
  
      ClearCLBuffer histogram = clke.createCLBuffer(new long[]{numberOfBins,1,1}, NativeTypeEnum.Float);
      Kernels.fillHistogram(clke, src, histogram, minimumGreyValue, maximumGreyValue);
      //releaseBuffers(args);
  
      //System.out.println("CL sum " + clke.op().sumPixels(histogram));
  
      // the histogram is written in args[1] which is supposed to be a one-dimensional image
      ImagePlus histogramImp = clke.convert(histogram, ImagePlus.class);
      histogram.close();
  
      // convert histogram
      float[] determinedHistogram = (float[])(histogramImp.getProcessor().getPixels());
      int[] convertedHistogram = new int[determinedHistogram.length];
  
      long sum = 0;
      for (int i = 0; i < determinedHistogram.length; i++) {
          convertedHistogram[i] = (int)determinedHistogram[i];
          sum += convertedHistogram[i];
      }
      //System.out.println("Sum: " + sum);
  
  
      String method = "Default";
  
      for (String choice : AutoThresholder.getMethods()) {
          if (choice.toLowerCase().compareTo(userSelectedMethod.toLowerCase()) == 0) {
              method = choice;
          }
      }
      //System.out.println("Method: " + method);
  
      float threshold = new AutoThresholder().getThreshold(method, convertedHistogram);
  
      // math source https://github.com/imagej/ImageJA/blob/master/src/main/java/ij/process/ImageProcessor.java#L692
      threshold = minimumGreyValue + ((threshold + 1.0f)/255.0f)*(maximumGreyValue-minimumGreyValue);
  
      //System.out.println("Threshold: " + threshold);
  
      Kernels.threshold(clke, src, dst, threshold);
  
      return true;
  }
  */

  public static boolean argMaximumZProjection(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst_max,
                                              ClearCLImage dst_arg)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("dst_arg", dst_arg);

    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "arg_max_project_3d_2d",
                        parameters);
  }

  public static boolean argMaximumZProjection(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst_max,
                                              ClearCLBuffer dst_arg)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("dst_arg", dst_arg);

    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "arg_max_project_3d_2d",
                        parameters);
  }

  public static boolean binaryAnd(CLKernelExecutor clke,
                                  ClearCLImage src1,
                                  ClearCLImage src2,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_and_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryAnd(CLKernelExecutor clke,
                                  ClearCLBuffer src1,
                                  ClearCLBuffer src2,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_and_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryXOr(CLKernelExecutor clke,
                                  ClearCLImage src1,
                                  ClearCLImage src2,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_xor_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryXOr(CLKernelExecutor clke,
                                  ClearCLBuffer src1,
                                  ClearCLBuffer src2,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_xor_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryNot(CLKernelExecutor clke,
                                  ClearCLImage src1,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_not_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryNot(CLKernelExecutor clke,
                                  ClearCLBuffer src1,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_not_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryOr(CLKernelExecutor clke,
                                 ClearCLImage src1,
                                 ClearCLImage src2,
                                 ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_or_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean binaryOr(CLKernelExecutor clke,
                                 ClearCLBuffer src1,
                                 ClearCLBuffer src2,
                                 ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "binary_or_" + src1.getDimension() + "d",
                        parameters);
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Float blurSigmaX,
                             Float blurSigmaY)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(0),
                                  blurSigmaX,
                                  blurSigmaY,
                                  0,
                                  src.getDimension());
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLBuffer dst,
                             Float blurSigmaX,
                             Float blurSigmaY)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(0),
                                  blurSigmaX,
                                  blurSigmaY,
                                  0,
                                  src.getDimension());
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Float blurSigmaX,
                             Float blurSigmaY)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(0),
                                  blurSigmaX,
                                  blurSigmaY,
                                  0,
                                  src.getDimension());
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Float blurSigmaX,
                             Float blurSigmaY,
                             Float blurSigmaZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(blurSigmaZ),
                                  blurSigmaX,
                                  blurSigmaY,
                                  blurSigmaZ,
                                  src.getDimension());
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLBuffer dst,
                             Float blurSigmaX,
                             Float blurSigmaY,
                             Float blurSigmaZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(blurSigmaZ),
                                  blurSigmaX,
                                  blurSigmaY,
                                  blurSigmaZ,
                                  src.getDimension());
  }

  public static boolean blur(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Float blurSigmaX,
                             Float blurSigmaY,
                             Float blurSigmaZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "blur.cl",
                                  "gaussian_blur_sep_image"
                                             + src.getDimension()
                                             + "d",
                                  sigmaToKernelSize(blurSigmaX),
                                  sigmaToKernelSize(blurSigmaY),
                                  sigmaToKernelSize(blurSigmaZ),
                                  blurSigmaX,
                                  blurSigmaY,
                                  blurSigmaZ,
                                  src.getDimension());
  }

  public static boolean countNonZeroPixelsLocally(CLKernelExecutor clke,
                                                  ClearCLBuffer src,
                                                  ClearCLBuffer dst,
                                                  Integer radiusX,
                                                  Integer radiusY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_image2d",
                        parameters);
  }

  public static boolean countNonZeroPixelsLocallySliceBySlice(CLKernelExecutor clke,
                                                              ClearCLBuffer src,
                                                              ClearCLBuffer dst,
                                                              Integer radiusX,
                                                              Integer radiusY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_slicewise_image3d",
                        parameters);
  }

  public static boolean countNonZeroVoxelsLocally(CLKernelExecutor clke,
                                                  ClearCLBuffer src,
                                                  ClearCLBuffer dst,
                                                  Integer radiusX,
                                                  Integer radiusY,
                                                  Integer radiusZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("Nz", radiusToKernelSize(radiusZ));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_image3d",
                        parameters);
  }

  public static boolean countNonZeroPixelsLocally(CLKernelExecutor clke,
                                                  ClearCLImage src,
                                                  ClearCLImage dst,
                                                  Integer radiusX,
                                                  Integer radiusY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_image2d",
                        parameters);
  }

  public static boolean countNonZeroPixelsLocallySliceBySlice(CLKernelExecutor clke,
                                                              ClearCLImage src,
                                                              ClearCLImage dst,
                                                              Integer radiusX,
                                                              Integer radiusY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_slicewise_image3d",
                        parameters);
  }

  public static boolean countNonZeroVoxelsLocally(CLKernelExecutor clke,
                                                  ClearCLImage src,
                                                  ClearCLImage dst,
                                                  Integer radiusX,
                                                  Integer radiusY,
                                                  Integer radiusZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("Nz", radiusToKernelSize(radiusZ));
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "binaryCounting.cl",
                        "count_nonzero_image3d",
                        parameters);
  }

  private static boolean executeSeparableKernel(CLKernelExecutor clke,
                                                Object src,
                                                Object dst,
                                                String clFilename,
                                                String kernelname,
                                                int kernelSizeX,
                                                int kernelSizeY,
                                                int kernelSizeZ,
                                                float blurSigmaX,
                                                float blurSigmaY,
                                                float blurSigmaZ,
                                                long dimensions)
  {
    int[] n = new int[]
    { kernelSizeX, kernelSizeY, kernelSizeZ };
    float[] blurSigma = new float[]
    { blurSigmaX, blurSigmaY, blurSigmaZ };

    Object temp;
    if (src instanceof ClearCLBuffer)
    {
      temp = clke.createCLBuffer((ClearCLBuffer) src);
    }
    else if (src instanceof ClearCLImage)
    {
      temp = clke.createCLImage((ClearCLImage) src);
    }
    else
    {
      throw new IllegalArgumentException("Error: Wrong type of images in blurFast");
    }

    HashMap<String, Object> parameters = new HashMap<>();

    if (blurSigma[0] > 0)
    {
      parameters.clear();
      parameters.put("N", n[0]);
      parameters.put("s", blurSigma[0]);
      parameters.put("dim", 0);
      parameters.put("src", src);
      if (dimensions == 2)
      {
        parameters.put("dst", temp);
      }
      else
      {
        parameters.put("dst", dst);
      }
      clke.execute(CLKernels.class,
                   clFilename,
                   kernelname,
                   parameters);
    }
    else
    {
      if (dimensions == 2)
      {
        Kernels.copyInternal(clke, src, temp, 2, 2);
      }
      else
      {
        Kernels.copyInternal(clke, src, dst, 3, 3);
      }
    }

    if (blurSigma[1] > 0)
    {
      parameters.clear();
      parameters.put("N", n[1]);
      parameters.put("s", blurSigma[1]);
      parameters.put("dim", 1);
      if (dimensions == 2)
      {
        parameters.put("src", temp);
        parameters.put("dst", dst);
      }
      else
      {
        parameters.put("src", dst);
        parameters.put("dst", temp);
      }
      clke.execute(CLKernels.class,
                   clFilename,
                   kernelname,
                   parameters);
    }
    else
    {
      if (dimensions == 2)
      {
        Kernels.copyInternal(clke, temp, dst, 2, 2);
      }
      else
      {
        Kernels.copyInternal(clke, dst, temp, 3, 3);
      }
    }

    if (dimensions == 3)
    {
      if (blurSigma[2] > 0)
      {
        parameters.clear();
        parameters.put("N", n[2]);
        parameters.put("s", blurSigma[2]);
        parameters.put("dim", 2);
        parameters.put("src", temp);
        parameters.put("dst", dst);
        clke.execute(CLKernels.class,
                     clFilename,
                     kernelname,
                     parameters);
      }
      else
      {
        Kernels.copyInternal(clke, temp, dst, 3, 3);
      }
    }

    if (temp instanceof ClearCLBuffer)
    {
      ((ClearCLBuffer) temp).close();
    }
    else if (temp instanceof ClearCLImage)
    {
      ((ClearCLImage) temp).close();
    }

    return true;
  }

  public static boolean blurSliceBySlice(CLKernelExecutor clke,
                                         ClearCLImage src,
                                         ClearCLImage dst,
                                         Integer kernelSizeX,
                                         Integer kernelSizeY,
                                         Float sigmaX,
                                         Float sigmaY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("sx", sigmaX);
    parameters.put("sy", sigmaY);
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "blur.cl",
                        "gaussian_blur_slicewise_image3d",
                        parameters);
  }

  public static boolean blurSliceBySlice(CLKernelExecutor clke,
                                         ClearCLBuffer src,
                                         ClearCLBuffer dst,
                                         int kernelSizeX,
                                         int kernelSizeY,
                                         float sigmaX,
                                         float sigmaY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("sx", sigmaX);
    parameters.put("sy", sigmaY);
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "blur.cl",
                        "gaussian_blur_slicewise_image3d",
                        parameters);
  }

  /*
  public static double[] centerOfMass(CLKernelExecutor clke, ClearCLBuffer input) {
      ClearCLBuffer multipliedWithCoordinate = clke.create(input.getDimensions(), NativeTypeEnum.Float);
      double sum = sumPixels(clke, input);
      double[] resultCenterOfMass;
      if (input.getDimension() > 2L && input.getDepth() > 1L) {
          resultCenterOfMass = new double[3];
      } else {
          resultCenterOfMass = new double[2];
      }
  
      multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 0);
      double sumX = sumPixels(clke, multipliedWithCoordinate);
      resultCenterOfMass[0] = sumX / sum;
      multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 1);
      double sumY = sumPixels(clke, multipliedWithCoordinate);
      resultCenterOfMass[1] = sumY / sum;
      if (input.getDimension() > 2L && input.getDepth() > 1L) {
          multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 2);
          double sumZ = sumPixels(clke, multipliedWithCoordinate);
          resultCenterOfMass[2] = sumZ / sum;
      }
  
      multipliedWithCoordinate.close();
      return resultCenterOfMass;
  }
  
  
  public static double[] centerOfMass(CLKernelExecutor clke, ClearCLImage input) {
      ClearCLImage multipliedWithCoordinate = clke.create(input.getDimensions(), ImageChannelDataType.Float);
      double sum = sumPixels(clke, input);
      double[] resultCenterOfMass;
      if (input.getDimension() > 2L && input.getDepth() > 1L) {
          resultCenterOfMass = new double[3];
      } else {
          resultCenterOfMass = new double[2];
      }
  
      multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 0);
      double sumX = sumPixels(clke, multipliedWithCoordinate);
      resultCenterOfMass[0] = sumX / sum;
      multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 1);
      double sumY = sumPixels(clke, multipliedWithCoordinate);
      resultCenterOfMass[1] = sumY / sum;
      if (input.getDimension() > 2L && input.getDepth() > 1L) {
          multiplyImageAndCoordinate(clke, input, multipliedWithCoordinate, 2);
          double sumZ = sumPixels(clke, multipliedWithCoordinate);
          resultCenterOfMass[2] = sumZ / sum;
      }
  
      multipliedWithCoordinate.close();
      return resultCenterOfMass;
  }
  */

  public static boolean copy(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLBuffer dst)
  {
    return copyInternal(clke,
                        src,
                        dst,
                        src.getDimension(),
                        dst.getDimension());
  }

  private static boolean copyInternal(CLKernelExecutor clke,
                                      Object src,
                                      Object dst,
                                      long srcNumberOfDimensions,
                                      long dstNumberOfDimensions)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(srcNumberOfDimensions,
                         dstNumberOfDimensions))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "duplication.cl",
                        "copy_" + srcNumberOfDimensions + "d",
                        parameters);
  }

  public static boolean copy(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLImage dst)
  {
    return copyInternal(clke,
                        src,
                        dst,
                        src.getDimension(),
                        dst.getDimension());
  }

  public static boolean copy(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst)
  {
    return copyInternal(clke,
                        src,
                        dst,
                        src.getDimension(),
                        dst.getDimension());
  }

  public static boolean copy(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst)
  {
    return copyInternal(clke,
                        src,
                        dst,
                        src.getDimension(),
                        dst.getDimension());
  }

  public static boolean copySlice(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer planeIndex)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("slice", planeIndex);
    if (src.getDimension() == 2 && dst.getDimension() == 3)
    {
      return clke.execute(CLKernels.class,
                          "duplication.cl",
                          "putSliceInStack",
                          parameters);
    }
    else if (src.getDimension() == 3 && dst.getDimension() == 2)
    {
      return clke.execute(CLKernels.class,
                          "duplication.cl",
                          "copySlice",
                          parameters);
    }
    else
    {
      throw new IllegalArgumentException("Images have wrong dimension. Must be 3D->2D or 2D->3D.");
    }
  }

  public static boolean copySlice(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer planeIndex)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("slice", planeIndex);
    // return clke.execute(CLKernels.class, "duplication.cl", "copySlice",
    // parameters);
    if (src.getDimension() == 2 && dst.getDimension() == 3)
    {
      return clke.execute(CLKernels.class,
                          "duplication.cl",
                          "putSliceInStack",
                          parameters);
    }
    else if (src.getDimension() == 3 && dst.getDimension() == 2)
    {
      return clke.execute(CLKernels.class,
                          "duplication.cl",
                          "copySlice",
                          parameters);
    }
    else
    {
      throw new IllegalArgumentException("Images have wrong dimension. Must be 3D->2D or 2D->3D.");
    }
  }

  public static boolean crop(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Integer startX,
                             Integer startY,
                             Integer startZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    parameters.put("start_z", startZ);
    return clke.execute(CLKernels.class,
                        "duplication.cl",
                        "crop_3d",
                        parameters);
  }

  public static boolean crop(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Integer startX,
                             Integer startY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    return clke.execute(CLKernels.class,
                        "duplication.cl",
                        "crop_2d",
                        parameters);
  }

  public static boolean crop(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Integer startX,
                             Integer startY,
                             Integer startZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    parameters.put("start_z", startZ);
    return clke.execute(CLKernels.class,
                        "duplication.cl",
                        "crop_3d",
                        parameters);
  }

  public static boolean crop(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Integer startX,
                             Integer startY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    return clke.execute(CLKernels.class,
                        "duplication.cl",
                        "crop_2d",
                        parameters);
  }

  public static boolean crossCorrelation(CLKernelExecutor clke,
                                         ClearCLBuffer src1,
                                         ClearCLBuffer meanSrc1,
                                         ClearCLBuffer src2,
                                         ClearCLBuffer meanSrc2,
                                         ClearCLBuffer dst,
                                         int radius,
                                         int deltaPos,
                                         int dimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("mean_src1", meanSrc1);
    parameters.put("src2", src2);
    parameters.put("mean_src2", meanSrc2);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("i", deltaPos);
    parameters.put("dimension", dimension);
    return clke.execute(CLKernels.class,
                        "cross_correlation.cl",
                        "cross_correlation_3d",
                        parameters);
  }

  public static boolean crossCorrelation(CLKernelExecutor clke,
                                         ClearCLImage src1,
                                         ClearCLImage meanSrc1,
                                         ClearCLImage src2,
                                         ClearCLImage meanSrc2,
                                         ClearCLImage dst,
                                         int radius,
                                         int deltaPos,
                                         int dimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("mean_src1", meanSrc1);
    parameters.put("src2", src2);
    parameters.put("mean_src2", meanSrc2);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("i", deltaPos);
    parameters.put("dimension", dimension);
    return clke.execute(CLKernels.class,
                        "cross_correlation.cl",
                        "cross_correlation_3d",
                        parameters);
  }

  public static boolean detectMaximaBox(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst,
                                        Integer radius)
  {
    return detectOptima(clke, src, dst, radius, true);
  }

  public static boolean detectMaximaBox(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst,
                                        Integer radius)
  {
    return detectOptima(clke, src, dst, radius, true);
  }

  public static boolean detectMaximaSliceBySliceBox(CLKernelExecutor clke,
                                                    ClearCLImage src,
                                                    ClearCLImage dst,
                                                    Integer radius)
  {
    return detectOptimaSliceBySlice(clke, src, dst, radius, true);
  }

  public static boolean detectMaximaSliceBySliceBox(CLKernelExecutor clke,
                                                    ClearCLBuffer src,
                                                    ClearCLBuffer dst,
                                                    Integer radius)
  {
    return detectOptimaSliceBySlice(clke, src, dst, radius, true);
  }

  public static boolean detectMinimaBox(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst,
                                        Integer radius)
  {
    return detectOptima(clke, src, dst, radius, false);
  }

  public static boolean detectMinimaBox(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst,
                                        Integer radius)
  {
    return detectOptima(clke, src, dst, radius, false);
  }

  public static boolean detectMinimaSliceBySliceBox(CLKernelExecutor clke,
                                                    ClearCLImage src,
                                                    ClearCLImage dst,
                                                    Integer radius)
  {
    return detectOptimaSliceBySlice(clke, src, dst, radius, false);
  }

  public static boolean detectMinimaSliceBySliceBox(CLKernelExecutor clke,
                                                    ClearCLBuffer src,
                                                    ClearCLBuffer dst,
                                                    Integer radius)
  {
    return detectOptimaSliceBySlice(clke, src, dst, radius, false);
  }

  public static boolean detectOptima(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     Integer radius,
                                     Boolean detectMaxima)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("detect_maxima", detectMaxima ? 1 : 0);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (detectOptima)");
    }
    return clke.execute(CLKernels.class,
                        "detection.cl",
                        "detect_local_optima_" + src.getDimension()
                                        + "d",
                        parameters);
  }

  public static boolean detectOptima(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     Integer radius,
                                     Boolean detectMaxima)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("detect_maxima", detectMaxima ? 1 : 0);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (detectOptima)");
    }
    return clke.execute(CLKernels.class,
                        "detection.cl",
                        "detect_local_optima_" + src.getDimension()
                                        + "d",
                        parameters);
  }

  public static boolean detectOptimaSliceBySlice(CLKernelExecutor clke,
                                                 ClearCLImage src,
                                                 ClearCLImage dst,
                                                 Integer radius,
                                                 Boolean detectMaxima)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("detect_maxima", detectMaxima ? 1 : 0);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (detectOptima)");
    }
    return clke.execute(CLKernels.class,
                        "detection.cl",
                        "detect_local_optima_" + src.getDimension()
                                        + "d_slice_by_slice",
                        parameters);
  }

  public static boolean detectOptimaSliceBySlice(CLKernelExecutor clke,
                                                 ClearCLBuffer src,
                                                 ClearCLBuffer dst,
                                                 Integer radius,
                                                 Boolean detectMaxima)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("detect_maxima", detectMaxima ? 1 : 0);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (detectOptima)");
    }
    return clke.execute(CLKernels.class,
                        "detection.cl",
                        "detect_local_optima_" + src.getDimension()
                                        + "d_slice_by_slice",
                        parameters);
  }

  public static boolean differenceOfGaussian(CLKernelExecutor clke,
                                             ClearCLImage src,
                                             ClearCLImage dst,
                                             Integer radius,
                                             Float sigmaMinuend,
                                             Float sigmaSubtrahend)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("sigma_minuend", sigmaMinuend);
    parameters.put("sigma_subtrahend", sigmaSubtrahend);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "differenceOfGaussian.cl",
                        "subtract_convolved_images_"
                                                   + src.getDimension()
                                                   + "d_fast",
                        parameters);
  }

  public static boolean differenceOfGaussianSliceBySlice(CLKernelExecutor clke,
                                                         ClearCLImage src,
                                                         ClearCLImage dst,
                                                         Integer radius,
                                                         Float sigmaMinuend,
                                                         Float sigmaSubtrahend)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);
    parameters.put("sigma_minuend", sigmaMinuend);
    parameters.put("sigma_subtrahend", sigmaSubtrahend);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "differenceOfGaussian.cl",
                        "subtract_convolved_images_"
                                                   + src.getDimension()
                                                   + "d_slice_by_slice",
                        parameters);
  }

  public static boolean dilateBox(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_box_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean dilateBox(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_box_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean dilateBoxSliceBySlice(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_box_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean dilateBoxSliceBySlice(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_box_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean dilateSphere(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_diamond_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean dilateSphere(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_diamond_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean dilateSphereSliceBySlice(CLKernelExecutor clke,
                                                 ClearCLImage src,
                                                 ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_diamond_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean dilateSphereSliceBySlice(CLKernelExecutor clke,
                                                 ClearCLBuffer src,
                                                 ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "dilate_diamond_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean divideImages(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage src1,
                                     ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "dividePixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean divideImages(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer src1,
                                     ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "dividePixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean downsample(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Float factorX,
                                   Float factorY,
                                   Float factorZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    parameters.put("factor_z", 1.f / factorZ);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_3d_nearest",
                        parameters);
  }

  public static boolean downsample(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Float factorX,
                                   Float factorY,
                                   Float factorZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    parameters.put("factor_z", 1.f / factorZ);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_3d_nearest",
                        parameters);
  }

  public static boolean downsample(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Float factorX,
                                   Float factorY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_2d_nearest",
                        parameters);
  }

  public static boolean downsample(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Float factorX,
                                   Float factorY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_2d_nearest",
                        parameters);
  }

  public static boolean downsampleSliceBySliceHalfMedian(CLKernelExecutor clke,
                                                         ClearCLImage src,
                                                         ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_xy_by_half_median",
                        parameters);
  }

  public static boolean downsampleSliceBySliceHalfMedian(CLKernelExecutor clke,
                                                         ClearCLBuffer src,
                                                         ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    return clke.execute(CLKernels.class,
                        "downsampling.cl",
                        "downsample_xy_by_half_median",
                        parameters);
  }

  public static boolean erodeSphere(CLKernelExecutor clke,
                                    ClearCLImage src,
                                    ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_diamond_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean erodeSphere(CLKernelExecutor clke,
                                    ClearCLBuffer src,
                                    ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_diamond_neighborhood_"
                                               + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean erodeSphereSliceBySlice(CLKernelExecutor clke,
                                                ClearCLImage src,
                                                ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_diamond_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean erodeSphereSliceBySlice(CLKernelExecutor clke,
                                                ClearCLBuffer src,
                                                ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_diamond_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean erodeBox(CLKernelExecutor clke,
                                 ClearCLImage src,
                                 ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_box_neighborhood_" + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean erodeBox(CLKernelExecutor clke,
                                 ClearCLBuffer src,
                                 ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_box_neighborhood_" + src.getDimension()
                                               + "d",
                        parameters);
  }

  public static boolean erodeBoxSliceBySlice(CLKernelExecutor clke,
                                             ClearCLImage src,
                                             ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_box_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean erodeBoxSliceBySlice(CLKernelExecutor clke,
                                             ClearCLBuffer src,
                                             ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    return clke.execute(CLKernels.class,
                        "binaryProcessing.cl",
                        "erode_box_neighborhood_slice_by_slice",
                        parameters);
  }

  public static boolean fillHistogram(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dstHistogram,
                                      Float minimumGreyValue,
                                      Float maximumGreyValue)
  {

    int stepSizeX = 1;
    int stepSizeY = 1;
    int stepSizeZ = 1;

    long[] globalSizes = new long[]
    { src.getHeight() / stepSizeZ, 1, 1 };

    long numberOfPartialHistograms = globalSizes[0] * globalSizes[1]
                                     * globalSizes[2];
    long[] histogramBufferSize = new long[]
    { dstHistogram.getWidth(), 1, numberOfPartialHistograms };

    long timeStamp = System.currentTimeMillis();

    // allocate memory for partial histograms
    ClearCLBuffer partialHistograms =
                                    clke.createCLBuffer(histogramBufferSize,
                                                        dstHistogram.getNativeType());

    //
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_histogram", partialHistograms);
    parameters.put("minimum", minimumGreyValue);
    parameters.put("maximum", maximumGreyValue);
    parameters.put("step_size_x", stepSizeX);
    parameters.put("step_size_y", stepSizeY);
    if (src.getDimension() > 2)
    {
      parameters.put("step_size_z", stepSizeZ);
    }
    clke.execute(CLKernels.class,
                 "histogram.cl",
                 "histogram_image_" + src.getDimension() + "d",
                 globalSizes,
                 parameters);

    Kernels.sumZProjection(clke, partialHistograms, dstHistogram);
    // IJ.log("Histogram generation took " + (System.currentTimeMillis() -
    // timeStamp) + " msec");

    partialHistograms.close();
    return true;
  }

  public static boolean gradientX(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientX_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean gradientY(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientY_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean gradientZ(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientZ_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean gradientX(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientX_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean gradientY(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientY_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean gradientZ(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    return clke.execute(CLKernels.class,
                        "neighbors.cl",
                        "gradientZ_" + src.getDimension() + "d",
                        parameters);
  }

  /*
  public static float[] histogram(CLKernelExecutor clke, ClearCLBuffer image, Float minGreyValue, Float maxGreyValue, Integer numberOfBins) {
      ClearCLBuffer histogram = clke.createCLBuffer(new long[]{numberOfBins, 1, 1}, NativeTypeEnum.Float);
  
      if (minGreyValue == null) {
          minGreyValue = new Double(Kernels.minimumOfAllPixels(clke, image)).floatValue();
      }
      if (maxGreyValue == null) {
          maxGreyValue = new Double(Kernels.maximumOfAllPixels(clke, image)).floatValue();
      }
  
      Kernels.fillHistogram(clke, image, histogram, minGreyValue, maxGreyValue);
  
      ImagePlus histogramImp = clke.convert(histogram, ImagePlus.class);
      histogram.close();
  
      float[] determinedHistogram = (float[])(histogramImp.getProcessor().getPixels());
      return determinedHistogram;
  }
  */

  public static boolean flip(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Boolean flipx,
                             Boolean flipy,
                             Boolean flipz)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    parameters.put("flipz", flipz ? 1 : 0);
    return clke.execute(CLKernels.class,
                        "flip.cl",
                        "flip_3d",
                        parameters);
  }

  public static boolean flip(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             Boolean flipx,
                             Boolean flipy)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    return clke.execute(CLKernels.class,
                        "flip.cl",
                        "flip_2d",
                        parameters);
  }

  public static boolean flip(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Boolean flipx,
                             Boolean flipy,
                             Boolean flipz)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    parameters.put("flipz", flipz ? 1 : 0);
    return clke.execute(CLKernels.class,
                        "flip.cl",
                        "flip_3d",
                        parameters);
  }

  public static boolean flip(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             Boolean flipx,
                             Boolean flipy)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    return clke.execute(CLKernels.class,
                        "flip.cl",
                        "flip_2d",
                        parameters);
  }

  public static boolean invert(CLKernelExecutor clke,
                               ClearCLImage input3d,
                               ClearCLImage output3d)
  {
    return multiplyImageAndScalar(clke, input3d, output3d, -1f);
  }

  public static boolean invert(CLKernelExecutor clke,
                               ClearCLBuffer input3d,
                               ClearCLBuffer output3d)
  {
    return multiplyImageAndScalar(clke, input3d, output3d, -1f);
  }

  public static boolean localThreshold(CLKernelExecutor clke,
                                       ClearCLImage src,
                                       ClearCLImage dst,
                                       ClearCLImage threshold)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("local_threshold", threshold);
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "thresholding.cl",
                        "apply_local_threshold_" + src.getDimension()
                                           + "d",
                        parameters);
  }

  public static boolean localThreshold(CLKernelExecutor clke,
                                       ClearCLBuffer src,
                                       ClearCLBuffer dst,
                                       ClearCLBuffer threshold)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("local_threshold", threshold);
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "thresholding.cl",
                        "apply_local_threshold_" + src.getDimension()
                                           + "d",
                        parameters);
  }

  public static boolean mask(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage mask,
                             ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (mask)");
    }
    return clke.execute(CLKernels.class,
                        "mask.cl",
                        "mask_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean mask(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer mask,
                             ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (mask)");
    }
    return clke.execute(CLKernels.class,
                        "mask.cl",
                        "mask_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean maskStackWithPlane(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage mask,
                                           ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "mask.cl",
                        "maskStackWithPlane",
                        parameters);
  }

  public static boolean maskStackWithPlane(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer mask,
                                           ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "mask.cl",
                        "maskStackWithPlane",
                        parameters);
  }

  public static boolean maximumSphere(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image2d",
                        parameters);
  }

  public static boolean maximumSphere(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image2d",
                        parameters);
  }

  public static boolean maximumSphere(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY,
                                      Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image3d",
                        parameters);
  }

  public static boolean maximumSphere(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY,
                                      Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image3d",
                        parameters);
  }

  @Deprecated
  public static boolean maximumIJ(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image2d_ij",
                        parameters);
  }

  @Deprecated
  public static boolean maximumIJ(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_image2d_ij",
                        parameters);
  }

  public static boolean maximumSliceBySliceSphere(CLKernelExecutor clke,
                                                  ClearCLImage src,
                                                  ClearCLImage dst,
                                                  Integer kernelSizeX,
                                                  Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_slicewise_image3d",
                        parameters);
  }

  public static boolean maximumBox(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   int radiusX,
                                   int radiusY,
                                   int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "max_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean maximumBox(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   int radiusX,
                                   int radiusY,
                                   int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "max_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean maximumSliceBySliceSphere(CLKernelExecutor clke,
                                                  ClearCLBuffer src,
                                                  ClearCLBuffer dst,
                                                  Integer kernelSizeX,
                                                  Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "maximum_slicewise_image3d",
                        parameters);
  }

  public static boolean maximumImages(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage src1,
                                      ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (maximumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "maxPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean maximumImages(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer src1,
                                      ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (maximumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "maxPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean maximumImageAndScalar(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst,
                                              Float valueB)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", valueB);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (maximumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "maxPixelwiseScalar_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean maximumImageAndScalar(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst,
                                              Float valueB)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", valueB);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (maximumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "maxPixelwiseScalar_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean minimumImages(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage src1,
                                      ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (minimumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "minPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean minimumImages(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer src1,
                                      ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (minimumImages)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "minPixelwise_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean minimumImageAndScalar(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst,
                                              Float valueB)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", valueB);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (minimumImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "minPixelwiseScalar_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean minimumImageAndScalar(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst,
                                              Float valueB)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", valueB);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (minimumImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "minPixelwiseScalar_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean maximumZProjection(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst_max)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "max_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean maximumZProjection(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst_max)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "max_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean minimumZProjection(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst_min)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_min", dst_min);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "min_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean minimumZProjection(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst_min)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_min", dst_min);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "min_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean meanZProjection(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "mean_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean meanZProjection(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "mean_project_3d_2d",
                 parameters);

    return true;
  }

  public static boolean maximumXYZProjection(CLKernelExecutor clke,
                                             ClearCLImage src,
                                             ClearCLImage dst_max,
                                             Integer projectedDimensionX,
                                             Integer projectedDimensionY,
                                             Integer projectedDimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("projection_x", projectedDimensionX);
    parameters.put("projection_y", projectedDimensionY);
    parameters.put("projection_dim", projectedDimension);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "max_project_dim_select_3d_2d",
                 parameters);

    return true;
  }

  public static boolean maximumXYZProjection(CLKernelExecutor clke,
                                             ClearCLBuffer src,
                                             ClearCLBuffer dst_max,
                                             Integer projectedDimensionX,
                                             Integer projectedDimensionY,
                                             Integer projectedDimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("projection_x", projectedDimensionX);
    parameters.put("projection_y", projectedDimensionY);
    parameters.put("projection_dim", projectedDimension);

    clke.execute(CLKernels.class,
                 "projections.cl",
                 "max_project_dim_select_3d_2d",
                 parameters);

    return true;
  }

  public static boolean meanSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image2d",
                        parameters);
  }

  public static boolean meanSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image2d",
                        parameters);
  }

  @Deprecated
  public static boolean meanIJ(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image2d_ij",
                        parameters);
  }

  @Deprecated
  public static boolean meanIJ(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image2d_ij",
                        parameters);
  }

  public static boolean meanSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image3d",
                        parameters);
  }

  public static boolean meanSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_image3d",
                        parameters);
  }

  public static boolean meanBox(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "mean_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean meanBox(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "mean_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean meanSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_slicewise_image3d",
                        parameters);
  }

  public static boolean meanSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "mean_slicewise_image3d",
                        parameters);
  }

  public static boolean medianSphere(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     Integer kernelSizeX,
                                     Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_image2d",
                        parameters);
  }

  public static boolean medianSphere(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     Integer kernelSizeX,
                                     Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_image2d",
                        parameters);
  }

  public static boolean medianSphere(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     Integer kernelSizeX,
                                     Integer kernelSizeY,
                                     Integer kernelSizeZ)
  {
    if (kernelSizeX * kernelSizeY
        * kernelSizeZ > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_image3d",
                        parameters);
  }

  public static boolean medianSphere(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     Integer kernelSizeX,
                                     Integer kernelSizeY,
                                     Integer kernelSizeZ)
  {
    if (kernelSizeX * kernelSizeY
        * kernelSizeZ > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_image3d",
                        parameters);
  }

  public static boolean medianSliceBySliceSphere(CLKernelExecutor clke,
                                                 ClearCLImage src,
                                                 ClearCLImage dst,
                                                 Integer kernelSizeX,
                                                 Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_slicewise_image3d",
                        parameters);
  }

  public static boolean medianSliceBySliceSphere(CLKernelExecutor clke,
                                                 ClearCLBuffer src,
                                                 ClearCLBuffer dst,
                                                 Integer kernelSizeX,
                                                 Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_slicewise_image3d",
                        parameters);
  }

  public static boolean medianBox(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_image2d",
                        parameters);
  }

  public static boolean medianBox(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_image2d",
                        parameters);
  }

  public static boolean medianBox(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY,
                                  Integer kernelSizeZ)
  {
    if (kernelSizeX * kernelSizeY
        * kernelSizeZ > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_image3d",
                        parameters);
  }

  public static boolean medianBox(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY,
                                  Integer kernelSizeZ)
  {
    if (kernelSizeX * kernelSizeY
        * kernelSizeZ > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_image3d",
                        parameters);
  }

  public static boolean medianSliceBySliceBox(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst,
                                              Integer kernelSizeX,
                                              Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_slicewise_image3d",
                        parameters);
  }

  public static boolean medianSliceBySliceBox(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst,
                                              Integer kernelSizeX,
                                              Integer kernelSizeY)
  {
    if (kernelSizeX * kernelSizeY > CLKernelExecutor.MAX_ARRAY_SIZE)
    {
      throw new IllegalArgumentException("Error: kernels of the medianSphere filter is too big. Consider increasing MAX_ARRAY_SIZE.");
    }
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "median_box_slicewise_image3d",
                        parameters);
  }

  public static boolean minimumSphere(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image2d",
                        parameters);
  }

  public static boolean minimumSphere(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image2d",
                        parameters);
  }

  public static boolean minimumSphere(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY,
                                      Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image3d",
                        parameters);
  }

  public static boolean minimumSphere(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY,
                                      Integer kernelSizeZ)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image3d",
                        parameters);
  }

  @Deprecated
  public static boolean minimumIJ(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image2d_ij",
                        parameters);
  }

  @Deprecated
  public static boolean minimumIJ(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer radius)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_image2d_ij",
                        parameters);
  }

  public static boolean minimumBox(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   int radiusX,
                                   int radiusY,
                                   int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "min_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean minimumBox(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   int radiusX,
                                   int radiusY,
                                   int radiusZ)
  {
    return executeSeparableKernel(clke,
                                  src,
                                  dst,
                                  "filtering.cl",
                                  "min_sep_image" + src.getDimension()
                                                  + "d",
                                  radiusToKernelSize(radiusX),
                                  radiusToKernelSize(radiusY),
                                  radiusToKernelSize(radiusZ),
                                  radiusX,
                                  radiusY,
                                  radiusZ,
                                  src.getDimension());
  }

  public static boolean minimumSliceBySliceSphere(CLKernelExecutor clke,
                                                  ClearCLImage src,
                                                  ClearCLImage dst,
                                                  Integer kernelSizeX,
                                                  Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_slicewise_image3d",
                        parameters);
  }

  public static boolean minimumSliceBySliceSphere(CLKernelExecutor clke,
                                                  ClearCLBuffer src,
                                                  ClearCLBuffer dst,
                                                  Integer kernelSizeX,
                                                  Integer kernelSizeY)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    return clke.execute(CLKernels.class,
                        "filtering.cl",
                        "minimum_slicewise_image3d",
                        parameters);
  }

  public static boolean multiplyImages(CLKernelExecutor clke,
                                       ClearCLImage src,
                                       ClearCLImage src1,
                                       ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyPixelwise_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean multiplyImages(CLKernelExecutor clke,
                                       ClearCLBuffer src,
                                       ClearCLBuffer src1,
                                       ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(),
                         src1.getDimension(),
                         dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyPixelwise_" + src.getDimension()
                                   + "d",
                        parameters);
  }

  public static boolean multiplyImageAndCoordinate(CLKernelExecutor clke,
                                                   ClearCLImage src,
                                                   ClearCLImage dst,
                                                   Integer dimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dimension", dimension);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (multiplyImageAndCoordinate)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiply_pixelwise_with_coordinate_3d",
                        parameters);
  }

  public static boolean multiplyImageAndCoordinate(CLKernelExecutor clke,
                                                   ClearCLBuffer src,
                                                   ClearCLBuffer dst,
                                                   Integer dimension)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dimension", dimension);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (multiplyImageAndCoordinate)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiply_pixelwise_with_coordinate_3d",
                        parameters);
  }

  public static boolean multiplyImageAndScalar(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Float scalar)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyScalar_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean multiplyImageAndScalar(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Float scalar)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyScalar_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean multiplySliceBySliceWithScalars(CLKernelExecutor clke,
                                                        ClearCLImage src,
                                                        ClearCLImage dst,
                                                        float[] scalars)
  {
    if (dst.getDimensions()[2] != scalars.length)
    {
      throw new IllegalArgumentException("Error: Wrong number of scalars in array.");
    }

    FloatBuffer buffer = FloatBuffer.allocate(scalars.length);
    buffer.put(scalars);

    ClearCLBuffer clBuffer = clke.createCLBuffer(new long[]
    { scalars.length }, NativeTypeEnum.Float);
    clBuffer.readFrom(buffer, true);
    buffer.clear();

    HashMap<String, Object> map = new HashMap<String, Object>();
    map.put("src", src);
    map.put("scalars", clBuffer);
    map.put("dst", dst);
    boolean result = clke.execute(CLKernels.class,
                                  "math.cl",
                                  "multiplySliceBySliceWithScalars",
                                  map);

    clBuffer.close();

    return result;
  }

  public static boolean multiplySliceBySliceWithScalars(CLKernelExecutor clke,
                                                        ClearCLBuffer src,
                                                        ClearCLBuffer dst,
                                                        float[] scalars)
  {
    if (dst.getDimensions()[2] != scalars.length)
    {
      throw new IllegalArgumentException("Error: Wrong number of scalars in array.");
    }

    FloatBuffer buffer = FloatBuffer.allocate(scalars.length);
    buffer.put(scalars);

    ClearCLBuffer clBuffer = clke.createCLBuffer(new long[]
    { scalars.length }, NativeTypeEnum.Float);
    clBuffer.readFrom(buffer, true);
    buffer.clear();

    HashMap<String, Object> map = new HashMap<String, Object>();
    map.put("src", src);
    map.put("scalars", clBuffer);
    map.put("dst", dst);
    boolean result = clke.execute(CLKernels.class,
                                  "math.cl",
                                  "multiplySliceBySliceWithScalars",
                                  map);

    clBuffer.close();

    return result;
  }

  public static boolean multiplyStackWithPlane(CLKernelExecutor clke,
                                               ClearCLImage input3d,
                                               ClearCLImage input2d,
                                               ClearCLImage output3d)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", input3d);
    parameters.put("src1", input2d);
    parameters.put("dst", output3d);
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyStackWithPlanePixelwise",
                        parameters);
  }

  public static boolean multiplyStackWithPlane(CLKernelExecutor clke,
                                               ClearCLBuffer input3d,
                                               ClearCLBuffer input2d,
                                               ClearCLBuffer output3d)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", input3d);
    parameters.put("src1", input2d);
    parameters.put("dst", output3d);
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "multiplyStackWithPlanePixelwise",
                        parameters);
  }

  public static boolean power(CLKernelExecutor clke,
                              ClearCLImage src,
                              ClearCLImage dst,
                              Float exponent)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("exponent", exponent);
    return clke.execute(CLKernels.class,
                        "math.cl",
                        "power_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean power(CLKernelExecutor clke,
                              ClearCLBuffer src,
                              ClearCLBuffer dst,
                              Float exponent)
  {

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("exponent", exponent);

    return clke.execute(CLKernels.class,
                        "math.cl",
                        "power_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean radialProjection(CLKernelExecutor clke,
                                         ClearCLImage src,
                                         ClearCLImage dst,
                                         Float deltaAngle)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("deltaAngle", deltaAngle);

    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "radialProjection3d",
                        parameters);
  }

  public static boolean radialProjection(CLKernelExecutor clke,
                                         ClearCLBuffer src,
                                         ClearCLBuffer dst,
                                         Float deltaAngle)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("deltaAngle", deltaAngle);

    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "radialProjection3d",
                        parameters);
  }

  public static boolean resliceBottom(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_bottom_3d",
                        parameters);
  }

  public static boolean resliceBottom(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_bottom_3d",
                        parameters);
  }

  public static boolean resliceLeft(CLKernelExecutor clke,
                                    ClearCLImage src,
                                    ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_left_3d",
                        parameters);
  }

  public static boolean resliceLeft(CLKernelExecutor clke,
                                    ClearCLBuffer src,
                                    ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_left_3d",
                        parameters);
  }

  public static boolean resliceRight(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_right_3d",
                        parameters);
  }

  public static boolean resliceRight(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_right_3d",
                        parameters);
  }

  public static boolean resliceTop(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_top_3d",
                        parameters);
  }

  public static boolean resliceTop(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "reslicing.cl",
                        "reslice_top_3d",
                        parameters);
  }

  public static boolean rotateLeft(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "rotate.cl",
                        "rotate_left_" + dst.getDimension() + "d",
                        parameters);
  }

  public static boolean rotateLeft(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "rotate.cl",
                        "rotate_left_" + dst.getDimension() + "d",
                        parameters);
  }

  public static boolean rotateRight(CLKernelExecutor clke,
                                    ClearCLBuffer src,
                                    ClearCLBuffer dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "rotate.cl",
                        "rotate_right_" + dst.getDimension() + "d",
                        parameters);
  }

  public static boolean rotateRight(CLKernelExecutor clke,
                                    ClearCLImage src,
                                    ClearCLImage dst)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    return clke.execute(CLKernels.class,
                        "rotate.cl",
                        "rotate_right_" + dst.getDimension() + "d",
                        parameters);
  }

  public static boolean set(CLKernelExecutor clke,
                            ClearCLImage clImage,
                            Float value)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("dst", clImage);
    parameters.put("value", value);

    return clke.execute(CLKernels.class,
                        "set.cl",
                        "set_" + clImage.getDimension() + "d",
                        parameters);
  }

  public static boolean set(CLKernelExecutor clke,
                            ClearCLBuffer clImage,
                            Float value)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("dst", clImage);
    parameters.put("value", value);

    return clke.execute(CLKernels.class,
                        "set.cl",
                        "set_" + clImage.getDimension() + "d",
                        parameters);
  }

  public static boolean splitStack(CLKernelExecutor clke,
                                   ClearCLImage clImageIn,
                                   ClearCLImage... clImagesOut)
  {
    if (clImagesOut.length > 12)
    {
      throw new IllegalArgumentException("Error: splitStack does not support more than 12 stacks.");
    }
    if (clImagesOut.length == 1)
    {
      return copy(clke, clImageIn, clImagesOut[0]);
    }
    if (clImagesOut.length == 0)
    {
      throw new IllegalArgumentException("Error: splitstack didn't get any output images.");
    }

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImageIn);
    for (int i = 0; i < clImagesOut.length; i++)
    {
      parameters.put("dst" + i, clImagesOut[i]);
    }

    return clke.execute(CLKernels.class,
                        "stacksplitting.cl",
                        "split_" + clImagesOut.length + "_stacks",
                        parameters);
  }

  public static boolean splitStack(CLKernelExecutor clke,
                                   ClearCLBuffer clImageIn,
                                   ClearCLBuffer... clImagesOut)
  {
    if (clImagesOut.length > 12)
    {
      throw new IllegalArgumentException("Error: splitStack does not support more than 12 stacks.");
    }
    if (clImagesOut.length == 1)
    {
      return copy(clke, clImageIn, clImagesOut[0]);
    }
    if (clImagesOut.length == 0)
    {
      throw new IllegalArgumentException("Error: splitstack didn't get any output images.");
    }

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImageIn);
    for (int i = 0; i < clImagesOut.length; i++)
    {
      parameters.put("dst" + i, clImagesOut[i]);
    }

    return clke.execute(CLKernels.class,
                        "stacksplitting.cl",
                        "split_" + clImagesOut.length + "_stacks",
                        parameters);
  }

  public static boolean subtractImages(CLKernelExecutor clke,
                                       ClearCLImage subtrahend,
                                       ClearCLImage minuend,
                                       ClearCLImage destination)
  {
    return addImagesWeighted(clke,
                             subtrahend,
                             minuend,
                             destination,
                             1f,
                             -1f);
  }

  public static boolean subtractImages(CLKernelExecutor clke,
                                       ClearCLBuffer subtrahend,
                                       ClearCLBuffer minuend,
                                       ClearCLBuffer destination)
  {
    return addImagesWeighted(clke,
                             subtrahend,
                             minuend,
                             destination,
                             1f,
                             -1f);
  }

  /*
    public static double maximumOfAllPixels(CLKernelExecutor clke, ClearCLImage clImage) {
        ClearCLImage clReducedImage = clImage;
        if (clImage.getDimension() == 3) {
            clReducedImage = clke.createCLImage(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getChannelDataType());
  
            HashMap<String, Object> parameters = new HashMap<>();
            parameters.put("src", clImage);
            parameters.put("dst_max", clReducedImage);
            clke.execute(CLKernels.class, "projections.cl", "max_project_3d_2d", parameters);
        }
  
        RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
        Cursor cursor = Views.iterable(rai).cursor();
        float maximumGreyValue = -Float.MAX_VALUE;
        while (cursor.hasNext()) {
            float greyValue = ((RealType) cursor.next()).getRealFloat();
            if (maximumGreyValue < greyValue) {
                maximumGreyValue = greyValue;
            }
        }
  
        if (clImage != clReducedImage) {
            clReducedImage.close();
        }
        return maximumGreyValue;
    }
  */

  /*
  public static double maximumOfAllPixels(CLKernelExecutor clke, ClearCLBuffer clImage) {
      ClearCLBuffer clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLBuffer(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getNativeType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst_max", clReducedImage);
          clke.execute(CLKernels.class, "projections.cl", "max_project_3d_2d", parameters);
      }
  
      RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
      Cursor cursor = Views.iterable(rai).cursor();
      float maximumGreyValue = -Float.MAX_VALUE;
      while (cursor.hasNext()) {
          float greyValue = ((RealType) cursor.next()).getRealFloat();
          if (maximumGreyValue < greyValue) {
              maximumGreyValue = greyValue;
          }
      }
  
      if (clImage != clReducedImage) {
          clReducedImage.close();
      }
      return maximumGreyValue;
  }
  
  
  public static double minimumOfAllPixels(CLKernelExecutor clke, ClearCLImage clImage) {
      ClearCLImage clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLImage(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getChannelDataType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst_min", clReducedImage);
          clke.execute(CLKernels.class, "projections.cl", "min_project_3d_2d", parameters);
      }
  
      RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
      Cursor cursor = Views.iterable(rai).cursor();
      float minimumGreyValue = Float.MAX_VALUE;
      while (cursor.hasNext()) {
          float greyValue = ((RealType) cursor.next()).getRealFloat();
          if (minimumGreyValue > greyValue) {
              minimumGreyValue = greyValue;
          }
      }
  
      if (clImage != clReducedImage) {
          clReducedImage.close();
      }
      return minimumGreyValue;
  }
  
  
  public static double minimumOfAllPixels(CLKernelExecutor clke, ClearCLBuffer clImage) {
      ClearCLBuffer clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLBuffer(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getNativeType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst_min", clReducedImage);
          clke.execute(CLKernels.class, "projections.cl", "min_project_3d_2d", parameters);
      }
  
      RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
      Cursor cursor = Views.iterable(rai).cursor();
      float minimumGreyValue = Float.MAX_VALUE;
      while (cursor.hasNext()) {
          float greyValue = ((RealType) cursor.next()).getRealFloat();
          if (minimumGreyValue > greyValue) {
              minimumGreyValue = greyValue;
          }
      }
  
      if (clImage != clReducedImage) {
          clReducedImage.close();
      }
      return minimumGreyValue;
  }
  
  public static double sumPixels(CLKernelExecutor clke, ClearCLImage clImage) {
      ClearCLImage clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLImage(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getChannelDataType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst", clReducedImage);
          clke.execute(CLKernels.class, "projections.cl", "sum_project_3d_2d", parameters);
      }
  
      RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
      Cursor cursor = Views.iterable(rai).cursor();
      float sum = 0;
      while (cursor.hasNext()) {
          sum += ((RealType) cursor.next()).getRealFloat();
      }
  
      if (clImage != clReducedImage) {
          clReducedImage.close();
      }
      return sum;
  }
  
  public static double sumPixels(CLKernelExecutor clke, ClearCLBuffer clImage) {
      ClearCLBuffer clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLBuffer(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getNativeType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst", clReducedImage);
          clke.execute(CLKernels.class, "projections.cl", "sum_project_3d_2d", parameters);
      }
  
      RandomAccessibleInterval rai = clke.convert(clReducedImage, RandomAccessibleInterval.class);
      Cursor cursor = Views.iterable(rai).cursor();
      float sum = 0;
      while (cursor.hasNext()) {
          sum += ((RealType) cursor.next()).getRealFloat();
      }
  
      if (clImage != clReducedImage) {
          clReducedImage.close();
      }
      return sum;
  }
  
  
  public static double[] sumPixelsSliceBySlice(CLKernelExecutor clke, ClearCLImage input) {
      if (input.getDimension() == 2) {
          return new double[]{sumPixels(clke, input)};
      }
  
      int numberOfImages = (int) input.getDepth();
      double[] result = new double[numberOfImages];
  
      ClearCLImage slice = clke.createCLImage(new long[]{input.getWidth(), input.getHeight()}, input.getChannelDataType());
      for (int z = 0; z < numberOfImages; z++) {
          copySlice(clke, input, slice, z);
          result[z] = sumPixels(clke, slice);
      }
      slice.close();
      return result;
  }
  
  public static double[] sumPixelsSliceBySlice(CLKernelExecutor clke, ClearCLBuffer input) {
      if (input.getDimension() == 2) {
          return new double[]{sumPixels(clke, input)};
      }
  
      int numberOfImages = (int) input.getDepth();
      double[] result = new double[numberOfImages];
  
      ClearCLBuffer slice = clke.createCLBuffer(new long[]{input.getWidth(), input.getHeight()}, input.getNativeType());
      for (int z = 0; z < numberOfImages; z++) {
          copySlice(clke, input, slice, z);
          result[z] = sumPixels(clke, slice);
      }
      slice.close();
      return result;
  }
  */

  public static boolean sumZProjection(CLKernelExecutor clke,
                                       ClearCLImage clImage,
                                       ClearCLImage clReducedImage)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImage);
    parameters.put("dst", clReducedImage);
    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "sum_project_3d_2d",
                        parameters);
  }

  public static boolean sumZProjection(CLKernelExecutor clke,
                                       ClearCLBuffer clImage,
                                       ClearCLBuffer clReducedImage)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImage);
    parameters.put("dst", clReducedImage);
    return clke.execute(CLKernels.class,
                        "projections.cl",
                        "sum_project_3d_2d",
                        parameters);
  }

  public static boolean tenengradWeightsSliceBySlice(CLKernelExecutor clke,
                                                     ClearCLImage clImageOut,
                                                     ClearCLImage clImageIn)
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImageIn);
    parameters.put("dst", clImageOut);
    return clke.execute(CLKernels.class,
                        "tenengradFusion.cl",
                        "tenengrad_weight_unnormalized_slice_wise",
                        parameters);
  }

  public static boolean tenengradFusion(CLKernelExecutor clke,
                                        ClearCLImage clImageOut,
                                        float[] blurSigmas,
                                        ClearCLImage... clImagesIn)
  {
    return tenengradFusion(clke,
                           clImageOut,
                           blurSigmas,
                           1.0f,
                           clImagesIn);
  }

  public static boolean tenengradFusion(CLKernelExecutor clke,
                                        ClearCLImage clImageOut,
                                        float[] blurSigmas,
                                        float exponent,
                                        ClearCLImage... clImagesIn)
  {
    if (clImagesIn.length > 12)
    {
      throw new IllegalArgumentException("Error: tenengradFusion does not support more than 12 stacks.");
    }
    if (clImagesIn.length == 1)
    {
      return copy(clke, clImagesIn[0], clImageOut);
    }
    if (clImagesIn.length == 0)
    {
      throw new IllegalArgumentException("Error: tenengradFusion didn't get any output images.");
    }
    if (!clImagesIn[0].isFloat())
    {
      System.out.println("Warning: tenengradFusion may only work on float images!");
    }

    HashMap<String, Object> lFusionParameters = new HashMap<>();

    ClearCLImage temporaryImage = clke.createCLImage(clImagesIn[0]);
    ClearCLImage temporaryImage2 = null;
    if (Math.abs(exponent - 1.0f) > 0.0001)
    {
      temporaryImage2 = clke.createCLImage(clImagesIn[0]);
    }

    ClearCLImage[] temporaryImages =
                                   new ClearCLImage[clImagesIn.length];
    for (int i = 0; i < clImagesIn.length; i++)
    {
      HashMap<String, Object> parameters = new HashMap<>();
      temporaryImages[i] = clke.createCLImage(clImagesIn[i]);
      parameters.put("src", clImagesIn[i]);
      parameters.put("dst", temporaryImage);

      clke.execute(CLKernels.class,
                   "tenengradFusion.cl",
                   "tenengrad_weight_unnormalized",
                   parameters);

      if (temporaryImage2 != null)
      {
        power(clke, temporaryImage, temporaryImage2, exponent);
        blur(clke,
             temporaryImage2,
             temporaryImages[i],
             blurSigmas[0],
             blurSigmas[1],
             blurSigmas[2]);
      }
      else
      {
        blur(clke,
             temporaryImage,
             temporaryImages[i],
             blurSigmas[0],
             blurSigmas[1],
             blurSigmas[2]);
      }

      lFusionParameters.put("src" + i, clImagesIn[i]);
      lFusionParameters.put("weight" + i, temporaryImages[i]);
    }

    lFusionParameters.put("dst", clImageOut);
    lFusionParameters.put("factor",
                          (int) (clImagesIn[0].getWidth()
                                 / temporaryImages[0].getWidth()));

    boolean success = clke.execute(CLKernels.class,
                                   "tenengradFusion.cl",
                                   String.format("tenengrad_fusion_with_provided_weights_%d_images",
                                                 clImagesIn.length),
                                   lFusionParameters);

    temporaryImage.close();
    for (int i = 0; i < temporaryImages.length; i++)
    {
      temporaryImages[i].close();
    }

    if (temporaryImage2 != null)
    {
      temporaryImage2.close();
    }

    return success;
  }

  public static boolean threshold(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Float threshold)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("threshold", threshold);
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "thresholding.cl",
                        "apply_threshold_" + src.getDimension() + "d",
                        parameters);
  }

  public static boolean threshold(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Float threshold)
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("threshold", threshold);
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    return clke.execute(CLKernels.class,
                        "thresholding.cl",
                        "apply_threshold_" + src.getDimension() + "d",
                        parameters);
  }

  private static boolean checkDimensions(long... numberOfDimensions)
  {
    for (int i = 0; i < numberOfDimensions.length - 1; i++)
    {
      if (!(numberOfDimensions[i] == numberOfDimensions[i + 1]))
      {
        return false;
      }
    }
    return true;
  }

}
