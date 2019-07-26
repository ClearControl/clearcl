package clearcl.ops.kernels;

import static clearcl.ops.kernels.KernelUtils.radiusToKernelSize;
import static clearcl.ops.kernels.KernelUtils.sigmaToKernelSize;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLHostImageBuffer;
import clearcl.ClearCLImage;
import clearcl.interfaces.ClearCLImageInterface;
import clearcl.ocllib.OCLlib;
import coremem.buffers.ContiguousBuffer;
import coremem.enums.NativeTypeEnum;
import coremem.offheap.OffHeapMemory;

/**
 * This class contains convenience access functions for OpenCL based image
 * processing.
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG
 * (http://mpi-cbg.de) March 2018
 * 
 * For documentation, see: https://clij.github.io/clij-docs/referenceJava Please
 * copy into javadoc here whenever you have a chance!
 */
public class Kernels
{

  /**
   * Computes the absolute value of every individual pixel x in a given image.
   * f(x) = |x|
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          - src image
   * @param dst
   *          - output image
   * @throws CLKernelException
   */
  public static void absolute(CLKernelExecutor clke,
                              ClearCLImageInterface src,
                              ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (absolute)");
    }

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "absolute_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * Calculates the sum of pairs of pixels x and y of two images X and Y. f(x,
   * y) = x + y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param src1
   *          second source image
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void addImages(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface src1,
                               ClearCLImageInterface dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "addPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void addImageAndScalar(CLKernelExecutor clke,
                                       ClearCLImageInterface src,
                                       ClearCLImageInterface dst,
                                       Float scalar) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "addScalar_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * Calculates the sum of pairs of pixels x and y from images X and Y weighted
   * with factors a and b. f(x, y, a, b) = x * a + y * b
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param src1
   *          second source image
   * @param dst
   *          output image
   * @param factor
   *          first factor (a)
   * @param factor1
   *          second factor (b)
   * @throws CLKernelException
   */
  public static void addImagesWeighted(CLKernelExecutor clke,
                                       ClearCLImageInterface src,
                                       ClearCLImageInterface src1,
                                       ClearCLImageInterface dst,
                                       Float factor,
                                       Float factor1) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "addWeightedPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void affineTransform(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     float[] matrix) throws CLKernelException
  {

    ClearCLBuffer matrixCl = clke.createCLBuffer(new long[]
    { matrix.length, 1, 1 }, NativeTypeEnum.Float);

    FloatBuffer buffer = FloatBuffer.wrap(matrix);
    matrixCl.readFrom(buffer, true);

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("input", src);
    parameters.put("output", dst);
    parameters.put("mat", matrixCl);

    clke.execute(OCLlib.class,
                 "kernels/affineTransforms.cl",
                 "affine",
                 parameters);

    matrixCl.close();

  }
  /*
    public static void affineTransform(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, AffineTransform3D at) {
        at = at.inverse();
        float[] matrix = AffineTransform.matrixToFloatArray(at);
        return affineTransform(clke, src, dst, matrix);
    }
    */

  public static void affineTransform(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     float[] matrix) throws CLKernelException
  {

    ClearCLBuffer matrixCl = clke.createCLBuffer(new long[]
    { matrix.length, 1, 1 }, NativeTypeEnum.Float);

    FloatBuffer buffer = FloatBuffer.wrap(matrix);
    matrixCl.readFrom(buffer, true);

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("input", src);
    parameters.put("output", dst);
    parameters.put("mat", matrixCl);

    clke.execute(OCLlib.class,
                 "kernels/affineTransforms_interpolate.cl",
                 "affine_interpolate",
                 parameters);

    matrixCl.close();

  }

  /*
  public static void affineTransform(CLKernelExecutor clke, ClearCLImage src, ClearCLImage dst, AffineTransform3D at) {
      at = at.inverse();
      float[] matrix = AffineTransform.matrixToFloatArray(at);
      return affineTransform(clke, src, dst, matrix);
  }
  */

  public static void applyVectorfield(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage vectorX,
                                      ClearCLImage vectorY,
                                      ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);

    clke.execute(OCLlib.class,
                 "kernels/deform_interpolate.cl",
                 "deform_2d_interpolate",
                 parameters);
  }

  public static void applyVectorfield(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage vectorX,
                                      ClearCLImage vectorY,
                                      ClearCLImage vectorZ,
                                      ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);
    parameters.put("vectorZ", vectorZ);

    clke.execute(OCLlib.class,
                 "kernels/deform_interpolate.cl",
                 "deform_3d_interpolate",
                 parameters);
  }

  public static void applyVectorfield(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer vectorX,
                                      ClearCLBuffer vectorY,
                                      ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);

    clke.execute(OCLlib.class,
                 "kernels/deform.cl",
                 "deform_2d",
                 parameters);
  }

  public static void applyVectorfield(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer vectorX,
                                      ClearCLBuffer vectorY,
                                      ClearCLBuffer vectorZ,
                                      ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("vectorX", vectorX);
    parameters.put("vectorY", vectorY);
    parameters.put("vectorZ", vectorZ);

    clke.execute(OCLlib.class,
                 "kernels/deform.cl",
                 "deform_3d",
                 parameters);
  }

  /*
  public static void automaticThreshold(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, String userSelectedMethod) {
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
  
  
  public static void automaticThreshold(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, String userSelectedMethod, Float minimumGreyValue, Float maximumGreyValue, Integer numberOfBins) {
  
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

  public static void argMaximumZProjection(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst_max,
                                           ClearCLImage dst_arg) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("dst_arg", dst_arg);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "arg_max_project_3d_2d",
                 parameters);
  }

  public static void argMaximumZProjection(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst_max,
                                           ClearCLBuffer dst_arg) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("dst_arg", dst_arg);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "arg_max_project_3d_2d",
                 parameters);
  }

  public static void binaryAnd(CLKernelExecutor clke,
                               ClearCLImage src1,
                               ClearCLImage src2,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_and_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryAnd(CLKernelExecutor clke,
                               ClearCLBuffer src1,
                               ClearCLBuffer src2,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_and_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryXOr(CLKernelExecutor clke,
                               ClearCLImage src1,
                               ClearCLImage src2,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_xor_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryXOr(CLKernelExecutor clke,
                               ClearCLBuffer src1,
                               ClearCLBuffer src2,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_xor_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryNot(CLKernelExecutor clke,
                               ClearCLImage src1,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_not_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryNot(CLKernelExecutor clke,
                               ClearCLBuffer src1,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_not_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryOr(CLKernelExecutor clke,
                              ClearCLImage src1,
                              ClearCLImage src2,
                              ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_or_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void binaryOr(CLKernelExecutor clke,
                              ClearCLBuffer src1,
                              ClearCLBuffer src2,
                              ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src1);
    parameters.put("src2", src2);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_or_" + src1.getDimension() + "d",
                 parameters);
  }

  public static void blur(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Float blurSigmaX,
                          Float blurSigmaY) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void blur(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLBuffer dst,
                          Float blurSigmaX,
                          Float blurSigmaY) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void blur(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Float blurSigmaX,
                          Float blurSigmaY) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void blur(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Float blurSigmaX,
                          Float blurSigmaY,
                          Float blurSigmaZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void blur(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLBuffer dst,
                          Float blurSigmaX,
                          Float blurSigmaY,
                          Float blurSigmaZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void blur(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Float blurSigmaX,
                          Float blurSigmaY,
                          Float blurSigmaZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "kernels/blur.cl",
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

  public static void countNonZeroPixelsLocally(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Integer radiusX,
                                               Integer radiusY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_image2d",
                 parameters);
  }

  public static void countNonZeroPixelsLocallySliceBySlice(CLKernelExecutor clke,
                                                           ClearCLBuffer src,
                                                           ClearCLBuffer dst,
                                                           Integer radiusX,
                                                           Integer radiusY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_slicewise_image3d",
                 parameters);
  }

  public static void countNonZeroVoxelsLocally(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Integer radiusX,
                                               Integer radiusY,
                                               Integer radiusZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("Nz", radiusToKernelSize(radiusZ));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_image3d",
                 parameters);
  }

  public static void countNonZeroPixelsLocally(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Integer radiusX,
                                               Integer radiusY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_image2d",
                 parameters);
  }

  public static void countNonZeroPixelsLocallySliceBySlice(CLKernelExecutor clke,
                                                           ClearCLImage src,
                                                           ClearCLImage dst,
                                                           Integer radiusX,
                                                           Integer radiusY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_slicewise_image3d",
                 parameters);
  }

  public static void countNonZeroVoxelsLocally(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Integer radiusX,
                                               Integer radiusY,
                                               Integer radiusZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", radiusToKernelSize(radiusX));
    parameters.put("Ny", radiusToKernelSize(radiusY));
    parameters.put("Nz", radiusToKernelSize(radiusZ));
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/binaryCounting.cl",
                 "count_nonzero_image3d",
                 parameters);
  }

  private static void executeSeparableKernel(CLKernelExecutor clke,
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
                                             long dimensions) throws CLKernelException
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

    try
    {
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
        clke.execute(OCLlib.class,
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
        clke.execute(OCLlib.class,
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
          clke.execute(OCLlib.class,
                       clFilename,
                       kernelname,
                       parameters);
        }
        else
        {
          Kernels.copyInternal(clke, temp, dst, 3, 3);
        }
      }
    }
    catch (CLKernelException clkExc)
    {
      throw clkExc;
    }
    finally
    {
      if (temp instanceof ClearCLBuffer)
      {
        ((ClearCLBuffer) temp).close();
      }
      else if (temp instanceof ClearCLImage)
      {
        ((ClearCLImage) temp).close();
      }
    }

  }

  public static void blurSliceBySlice(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Integer kernelSizeX,
                                      Integer kernelSizeY,
                                      Float sigmaX,
                                      Float sigmaY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("sx", sigmaX);
    parameters.put("sy", sigmaY);
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/blur.cl",
                 "gaussian_blur_slicewise_image3d",
                 parameters);
  }

  public static void blurSliceBySlice(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      int kernelSizeX,
                                      int kernelSizeY,
                                      float sigmaX,
                                      float sigmaY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("sx", sigmaX);
    parameters.put("sy", sigmaY);
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/blur.cl",
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

  public static void copy(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLBuffer dst) throws CLKernelException
  {
    copyInternal(clke,
                 src,
                 dst,
                 src.getDimension(),
                 dst.getDimension());
  }

  private static void copyInternal(CLKernelExecutor clke,
                                   Object src,
                                   Object dst,
                                   long srcNumberOfDimensions,
                                   long dstNumberOfDimensions) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(srcNumberOfDimensions,
                         dstNumberOfDimensions))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/duplication.cl",
                 "copy_" + srcNumberOfDimensions + "d",
                 parameters);
  }

  public static void copy(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLImage dst) throws CLKernelException
  {
    copyInternal(clke,
                 src,
                 dst,
                 src.getDimension(),
                 dst.getDimension());
  }

  public static void copy(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst) throws CLKernelException
  {
    copyInternal(clke,
                 src,
                 dst,
                 src.getDimension(),
                 dst.getDimension());
  }

  public static void copy(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst) throws CLKernelException
  {
    copyInternal(clke,
                 src,
                 dst,
                 src.getDimension(),
                 dst.getDimension());
  }

  public static void copySlice(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer planeIndex) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("slice", planeIndex);
    if (src.getDimension() == 2 && dst.getDimension() == 3)
    {
      clke.execute(OCLlib.class,
                   "kernels/duplication.cl",
                   "putSliceInStack",
                   parameters);
    }
    else if (src.getDimension() == 3 && dst.getDimension() == 2)
    {
      clke.execute(OCLlib.class,
                   "kernels/duplication.cl",
                   "copySlice",
                   parameters);
    }
    else
    {
      throw new IllegalArgumentException("Images have wrong dimension. Must be 3D->2D or 2D->3D.");
    }
  }

  public static void copySlice(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer planeIndex) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("slice", planeIndex);
    // clke.execute(OCLlib.class, "duplication.cl", "copySlice",
    // parameters);
    if (src.getDimension() == 2 && dst.getDimension() == 3)
    {
      clke.execute(OCLlib.class,
                   "kernels/duplication.cl",
                   "putSliceInStack",
                   parameters);
    }
    else if (src.getDimension() == 3 && dst.getDimension() == 2)
    {
      clke.execute(OCLlib.class,
                   "kernels/duplication.cl",
                   "copySlice",
                   parameters);
    }
    else
    {
      throw new IllegalArgumentException("Images have wrong dimension. Must be 3D->2D or 2D->3D.");
    }
  }

  public static void crop(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Integer startX,
                          Integer startY,
                          Integer startZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    parameters.put("start_z", startZ);
    clke.execute(OCLlib.class,
                 "kernels/duplication.cl",
                 "crop_3d",
                 parameters);
  }

  public static void crop(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Integer startX,
                          Integer startY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    clke.execute(OCLlib.class,
                 "kernels/duplication.cl",
                 "crop_2d",
                 parameters);
  }

  public static void crop(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Integer startX,
                          Integer startY,
                          Integer startZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    parameters.put("start_z", startZ);
    clke.execute(OCLlib.class,
                 "kernels/duplication.cl",
                 "crop_3d",
                 parameters);
  }

  public static void crop(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Integer startX,
                          Integer startY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("start_x", startX);
    parameters.put("start_y", startY);
    clke.execute(OCLlib.class,
                 "kernels/duplication.cl",
                 "crop_2d",
                 parameters);
  }

  public static void crossCorrelation(CLKernelExecutor clke,
                                      ClearCLBuffer src1,
                                      ClearCLBuffer meanSrc1,
                                      ClearCLBuffer src2,
                                      ClearCLBuffer meanSrc2,
                                      ClearCLBuffer dst,
                                      int radius,
                                      int deltaPos,
                                      int dimension) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/cross_correlation.cl",
                 "cross_correlation_3d",
                 parameters);
  }

  public static void crossCorrelation(CLKernelExecutor clke,
                                      ClearCLImage src1,
                                      ClearCLImage meanSrc1,
                                      ClearCLImage src2,
                                      ClearCLImage meanSrc2,
                                      ClearCLImage dst,
                                      int radius,
                                      int deltaPos,
                                      int dimension) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/cross_correlation.cl",
                 "cross_correlation_3d",
                 parameters);
  }

  public static void detectMaximaBox(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, true);
  }

  public static void detectMaximaBox(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, true);
  }

  public static void detectMaximaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLImage src,
                                                 ClearCLImage dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, true);
  }

  public static void detectMaximaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLBuffer src,
                                                 ClearCLBuffer dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, true);
  }

  public static void detectMinimaBox(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, false);
  }

  public static void detectMinimaBox(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, false);
  }

  public static void detectMinimaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLImage src,
                                                 ClearCLImage dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, false);
  }

  public static void detectMinimaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLBuffer src,
                                                 ClearCLBuffer dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, false);
  }

  public static void detectOptima(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer radius,
                                  Boolean detectMaxima) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/detection.cl",
                 "detect_local_optima_" + src.getDimension() + "d",
                 parameters);
  }

  public static void detectOptima(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer radius,
                                  Boolean detectMaxima) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/detection.cl",
                 "detect_local_optima_" + src.getDimension() + "d",
                 parameters);
  }

  public static void detectOptimaSliceBySlice(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst,
                                              Integer radius,
                                              Boolean detectMaxima) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/detection.cl",
                 "detect_local_optima_" + src.getDimension()
                                         + "d_slice_by_slice",
                 parameters);
  }

  public static void detectOptimaSliceBySlice(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst,
                                              Integer radius,
                                              Boolean detectMaxima) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/detection.cl",
                 "detect_local_optima_" + src.getDimension()
                                         + "d_slice_by_slice",
                 parameters);
  }

  public static void differenceOfGaussian(CLKernelExecutor clke,
                                          ClearCLImage src,
                                          ClearCLImage dst,
                                          Integer radius,
                                          Float sigmaMinuend,
                                          Float sigmaSubtrahend) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/differenceOfGaussian.cl",
                 "subtract_convolved_images_" + src.getDimension()
                                                    + "d_fast",
                 parameters);
  }

  public static void differenceOfGaussianSliceBySlice(CLKernelExecutor clke,
                                                      ClearCLImage src,
                                                      ClearCLImage dst,
                                                      Integer radius,
                                                      Float sigmaMinuend,
                                                      Float sigmaSubtrahend) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/differenceOfGaussian.cl",
                 "subtract_convolved_images_" + src.getDimension()
                                                    + "d_slice_by_slice",
                 parameters);
  }

  public static void dilateBox(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_box_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void dilateBox(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_box_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void dilateBoxSliceBySlice(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_box_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void dilateBoxSliceBySlice(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_box_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void dilateSphere(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_diamond_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void dilateSphere(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_diamond_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void dilateSphereSliceBySlice(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_diamond_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void dilateSphereSliceBySlice(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "dilate_diamond_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void divideImages(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage src1,
                                  ClearCLImage dst) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "dividePixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void divideImages(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer src1,
                                  ClearCLBuffer dst) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "dividePixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void downsample(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                Float factorX,
                                Float factorY,
                                Float factorZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    parameters.put("factor_z", 1.f / factorZ);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_3d_nearest",
                 parameters);
  }

  public static void downsample(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                Float factorX,
                                Float factorY,
                                Float factorZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    parameters.put("factor_z", 1.f / factorZ);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_3d_nearest",
                 parameters);
  }

  public static void downsample(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                Float factorX,
                                Float factorY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_2d_nearest",
                 parameters);
  }

  public static void downsample(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                Float factorX,
                                Float factorY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("factor_x", 1.f / factorX);
    parameters.put("factor_y", 1.f / factorY);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_2d_nearest",
                 parameters);
  }

  public static void downsampleSliceBySliceHalfMedian(CLKernelExecutor clke,
                                                      ClearCLImage src,
                                                      ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_xy_by_half_median",
                 parameters);
  }

  public static void downsampleSliceBySliceHalfMedian(CLKernelExecutor clke,
                                                      ClearCLBuffer src,
                                                      ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    clke.execute(OCLlib.class,
                 "kernels/downsampling.cl",
                 "downsample_xy_by_half_median",
                 parameters);
  }

  public static void erodeSphere(CLKernelExecutor clke,
                                 ClearCLImage src,
                                 ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_diamond_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void erodeSphere(CLKernelExecutor clke,
                                 ClearCLBuffer src,
                                 ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_diamond_neighborhood_" + src.getDimension()
                                                + "d",
                 parameters);
  }

  public static void erodeSphereSliceBySlice(CLKernelExecutor clke,
                                             ClearCLImage src,
                                             ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_diamond_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void erodeSphereSliceBySlice(CLKernelExecutor clke,
                                             ClearCLBuffer src,
                                             ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_diamond_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void erodeBox(CLKernelExecutor clke,
                              ClearCLImage src,
                              ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_box_neighborhood_" + src.getDimension() + "d",
                 parameters);
  }

  public static void erodeBox(CLKernelExecutor clke,
                              ClearCLBuffer src,
                              ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_box_neighborhood_" + src.getDimension() + "d",
                 parameters);
  }

  public static void erodeBoxSliceBySlice(CLKernelExecutor clke,
                                          ClearCLImage src,
                                          ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_box_neighborhood_slice_by_slice",
                 parameters);
  }

  public static void erodeBoxSliceBySlice(CLKernelExecutor clke,
                                          ClearCLBuffer src,
                                          ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "erode_box_neighborhood_slice_by_slice",
                 parameters);
  }

  /**
   * Calculates a histogram from the input Buffer, and places the histogram
   * values in the dstHistogram, Create the dstHistogram as follows: long[]
   * histDims = {numberOfBins,1,1}; ClearCLBuffer histogram =
   * clke.createCLBuffer(histDims, NativeTypeEnum.Float);
   * 
   * @param clke
   *          CLKernelExecutor instance *
   * @param src
   *          Input CLBuffer
   * @param dstHistogram
   *          output histogram
   * @param minimumGreyValue
   *          minimum value of the input image
   * @param maximumGreyValue
   *          maximum value of the input image
   * @throws CLKernelException
   */
  public static void histogram(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLBuffer dstHistogram,
                               Float minimumGreyValue,
                               Float maximumGreyValue) throws CLKernelException
  {

    int stepSizeX = 1;
    int stepSizeY = 1;
    int stepSizeZ = 1;

    long[] globalSizes = new long[]
    { src.getWidth() / stepSizeZ, 1, 1 };

    long numberOfPartialHistograms = globalSizes[0] * globalSizes[1]
                                     * globalSizes[2];
    long[] histogramBufferSize = new long[]
    { dstHistogram.getWidth(), 1, numberOfPartialHistograms };

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
    clke.execute(OCLlib.class,
                 "kernels/histogram.cl",
                 "histogram_image_" + src.getDimension() + "d",
                 globalSizes,
                 parameters);

    Kernels.sumZProjection(clke, partialHistograms, dstHistogram);
    // IJ.log("Histogram generation took " + (System.currentTimeMillis() -
    // timeStamp) + " msec");

    partialHistograms.close();
  }

  /**
   * Calculates a histogram from the input Buffer, and places the histogram
   * values in the dstHistogram. Size (number of bins) of the histogram is
   * determined by the size of dstHistogra,
   * 
   * long[] histDims = {numberOfBins,1,1}; ClearCLBuffer histogram =
   * clke.createCLBuffer(histDims, NativeTypeEnum.Float);
   * 
   * @param clke
   *          CLKernelExecutor instance *
   * @param src
   *          Input CLBuffer
   * @param dstHistogram
   *          output histogram Need to be allocated and not null. Size
   *          determines histogram size
   * @param minimumGreyValue
   *          minimum value of the input image
   * @param maximumGreyValue
   *          maximum value of the input image
   * @throws CLKernelException
   */
  public static void histogram(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               int[] dstHistogram,
                               Float minimumGreyValue,
                               Float maximumGreyValue) throws CLKernelException
  {
    long[] histDims =
    { dstHistogram.length, 1, 1 };
    // TODO: It must be more efficient to do this with Int or Long,
    // however, CLKernelExecutor does not support those types
    ClearCLBuffer histogram =
                            clke.createCLBuffer(histDims,
                                                NativeTypeEnum.Float);
    histogram(clke,
              src,
              histogram,
              minimumGreyValue,
              maximumGreyValue);

    OffHeapMemory lBuffer =
                          OffHeapMemory.allocateFloats(dstHistogram.length);
    histogram.writeTo(lBuffer, true);
    float[] fBuf = new float[dstHistogram.length];
    lBuffer.copyTo(fBuf);
    for (int i = 0; i < fBuf.length; i++)
    {
      dstHistogram[i] = (int) fBuf[i];
    }
  }

  public static void gradientX(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
                 "gradientX_" + src.getDimension() + "d",
                 parameters);
  }

  public static void gradientY(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
                 "gradientY_" + src.getDimension() + "d",
                 parameters);
  }

  public static void gradientZ(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
                 "gradientZ_" + src.getDimension() + "d",
                 parameters);
  }

  public static void gradientX(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
                 "gradientX_" + src.getDimension() + "d",
                 parameters);
  }

  public static void gradientY(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
                 "gradientY_" + src.getDimension() + "d",
                 parameters);
  }

  public static void gradientZ(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (copy)");
    }
    clke.execute(OCLlib.class,
                 "kernels/neighbors.cl",
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
      determinedHistogram;
  }
  */

  public static void flip(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Boolean flipx,
                          Boolean flipy,
                          Boolean flipz) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    parameters.put("flipz", flipz ? 1 : 0);
    clke.execute(OCLlib.class,
                 "kernels/flip.cl",
                 "flip_3d",
                 parameters);
  }

  public static void flip(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage dst,
                          Boolean flipx,
                          Boolean flipy) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    clke.execute(OCLlib.class,
                 "kernels/flip.cl",
                 "flip_2d",
                 parameters);
  }

  public static void flip(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Boolean flipx,
                          Boolean flipy,
                          Boolean flipz) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    parameters.put("flipz", flipz ? 1 : 0);
    clke.execute(OCLlib.class,
                 "kernels/flip.cl",
                 "flip_3d",
                 parameters);
  }

  public static void flip(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer dst,
                          Boolean flipx,
                          Boolean flipy) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("flipx", flipx ? 1 : 0);
    parameters.put("flipy", flipy ? 1 : 0);
    clke.execute(OCLlib.class,
                 "kernels/flip.cl",
                 "flip_2d",
                 parameters);
  }

  public static void invert(CLKernelExecutor clke,
                            ClearCLImage input3d,
                            ClearCLImage output3d) throws CLKernelException
  {
    multiplyImageAndScalar(clke, input3d, output3d, -1f);
  }

  public static void invert(CLKernelExecutor clke,
                            ClearCLBuffer input3d,
                            ClearCLBuffer output3d) throws CLKernelException
  {
    multiplyImageAndScalar(clke, input3d, output3d, -1f);
  }

  public static void localThreshold(CLKernelExecutor clke,
                                    ClearCLImage src,
                                    ClearCLImage dst,
                                    ClearCLImage threshold) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/thresholding.cl",
                 "apply_local_threshold_" + src.getDimension() + "d",
                 parameters);
  }

  public static void localThreshold(CLKernelExecutor clke,
                                    ClearCLBuffer src,
                                    ClearCLBuffer dst,
                                    ClearCLBuffer threshold) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/thresholding.cl",
                 "apply_local_threshold_" + src.getDimension() + "d",
                 parameters);
  }

  public static void mask(CLKernelExecutor clke,
                          ClearCLImage src,
                          ClearCLImage mask,
                          ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (mask)");
    }
    clke.execute(OCLlib.class,
                 "kernels/mask.cl",
                 "mask_" + src.getDimension() + "d",
                 parameters);
  }

  public static void mask(CLKernelExecutor clke,
                          ClearCLBuffer src,
                          ClearCLBuffer mask,
                          ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (mask)");
    }
    clke.execute(OCLlib.class,
                 "kernels/mask.cl",
                 "mask_" + src.getDimension() + "d",
                 parameters);
  }

  public static void maskStackWithPlane(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage mask,
                                        ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/mask.cl",
                 "maskStackWithPlane",
                 parameters);
  }

  public static void maskStackWithPlane(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer mask,
                                        ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("mask", mask);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/mask.cl",
                 "maskStackWithPlane",
                 parameters);
  }

  public static void maximumSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image2d",
                 parameters);
  }

  public static void maximumSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image2d",
                 parameters);
  }

  public static void maximumSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image3d",
                 parameters);
  }

  public static void maximumSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image3d",
                 parameters);
  }

  @Deprecated
  public static void maximumIJ(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image2d_ij",
                 parameters);
  }

  @Deprecated
  public static void maximumIJ(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_image2d_ij",
                 parameters);
  }

  public static void maximumSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_slicewise_image3d",
                 parameters);
  }

  public static void maximumBox(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "filtering.cl",
                           "max_sep_image" + src.getDimension() + "d",
                           radiusToKernelSize(radiusX),
                           radiusToKernelSize(radiusY),
                           radiusToKernelSize(radiusZ),
                           radiusX,
                           radiusY,
                           radiusZ,
                           src.getDimension());
  }

  public static void maximumBox(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "filtering.cl",
                           "max_sep_image" + src.getDimension() + "d",
                           radiusToKernelSize(radiusX),
                           radiusToKernelSize(radiusY),
                           radiusToKernelSize(radiusZ),
                           radiusX,
                           radiusY,
                           radiusZ,
                           src.getDimension());
  }

  public static void maximumSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "maximum_slicewise_image3d",
                 parameters);
  }

  public static void maximumImages(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage src1,
                                   ClearCLImage dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "maxPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void maximumImages(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer src1,
                                   ClearCLBuffer dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "maxPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void maximumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst,
                                           Float valueB) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "maxPixelwiseScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void maximumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst,
                                           Float valueB) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "maxPixelwiseScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void minimumImages(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage src1,
                                   ClearCLImage dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "minPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void minimumImages(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer src1,
                                   ClearCLBuffer dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "minPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void minimumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst,
                                           Float valueB) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "minPixelwiseScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void minimumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst,
                                           Float valueB) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "minPixelwiseScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void maximumZProjection(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst_max) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "max_project_3d_2d",
                 parameters);

  }

  public static void maximumZProjection(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst_max) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "max_project_3d_2d",
                 parameters);
  }

  public static void minimumZProjection(CLKernelExecutor clke,
                                        ClearCLImage src,
                                        ClearCLImage dst_min) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_min", dst_min);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "min_project_3d_2d",
                 parameters);
  }

  public static void minimumZProjection(CLKernelExecutor clke,
                                        ClearCLBuffer src,
                                        ClearCLBuffer dst_min) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_min", dst_min);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "min_project_3d_2d",
                 parameters);
  }

  public static float[] minMax(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               int nrReductions) throws CLKernelException
  {

    ClearCLBuffer mScratchBuffer = clke.createCLBuffer(new long[]
    { 2 * nrReductions }, src.getNativeType());

    ClearCLHostImageBuffer mScratchHostBuffer =
                                              ClearCLHostImageBuffer.allocateSameAs(mScratchBuffer);

    long size = src.getWidth() * src.getHeight()
                * src.getNumberOfChannels();
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", mScratchBuffer);
    parameters.put("length", size);

    clke.execute(OCLlib.class,
                 "kernels/reductions.cl",
                 "reduce_minmax_" + src.getDimension() + "d",
                 new long[]
                 { Math.min(size, nrReductions) }, parameters);

    mScratchBuffer.copyTo(mScratchHostBuffer, true);

    ContiguousBuffer lContiguousBuffer =
                                       ContiguousBuffer.wrap(mScratchHostBuffer.getContiguousMemory());

    float lMin = Float.POSITIVE_INFINITY;
    float lMax = Float.NEGATIVE_INFINITY;
    lContiguousBuffer.rewind();
    if (null == src.getNativeType())
    {
      throw new CLKernelException("minmax only support data of type float, unsigned short, and unsigned byte");
    }
    else
      switch (src.getNativeType())
      {
      case Float:
        while (lContiguousBuffer.hasRemainingFloat())
        {
          float lMinValue = lContiguousBuffer.readFloat();
          lMin = Math.min(lMin, lMinValue);
          float lMaxValue = lContiguousBuffer.readFloat();
          lMax = Math.max(lMax, lMaxValue);
        }
        break;
      case UnsignedShort:
        while (lContiguousBuffer.hasRemainingShort())
        {
          int lMinValue = lContiguousBuffer.readShort();
          lMin = Math.min(0xFFFF & (short) lMinValue, lMin);
          int lMaxValue = lContiguousBuffer.readShort();
          lMax = Math.max(0xFFFF & (short) lMaxValue, lMax);
        }
        break;
      case UnsignedByte:
        while (lContiguousBuffer.hasRemainingByte())
        {
          int lMinValue = lContiguousBuffer.readByte();
          lMin = Math.min(0xFF & (byte) lMinValue, lMin);
          int lMaxValue = lContiguousBuffer.readByte();
          lMax = Math.max(0xFF & (byte) lMaxValue, lMax);
        }
        break;
      default:
        throw new CLKernelException("minmax only support data of type float, unsigned short, and unsigned byte");
      }

    return new float[]
    { lMin, lMax };

  }

  public static void meanZProjection(CLKernelExecutor clke,
                                     ClearCLImage src,
                                     ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "mean_project_3d_2d",
                 parameters);
  }

  public static void meanZProjection(CLKernelExecutor clke,
                                     ClearCLBuffer src,
                                     ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "mean_project_3d_2d",
                 parameters);
  }

  public static void maximumXYZProjection(CLKernelExecutor clke,
                                          ClearCLImage src,
                                          ClearCLImage dst_max,
                                          Integer projectedDimensionX,
                                          Integer projectedDimensionY,
                                          Integer projectedDimension) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("projection_x", projectedDimensionX);
    parameters.put("projection_y", projectedDimensionY);
    parameters.put("projection_dim", projectedDimension);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "max_project_dim_select_3d_2d",
                 parameters);
  }

  public static void maximumXYZProjection(CLKernelExecutor clke,
                                          ClearCLBuffer src,
                                          ClearCLBuffer dst_max,
                                          Integer projectedDimensionX,
                                          Integer projectedDimensionY,
                                          Integer projectedDimension) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst_max);
    parameters.put("projection_x", projectedDimensionX);
    parameters.put("projection_y", projectedDimensionY);
    parameters.put("projection_dim", projectedDimension);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "max_project_dim_select_3d_2d",
                 parameters);
  }

  public static void meanSphere(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                Integer kernelSizeX,
                                Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image2d",
                 parameters);
  }

  public static void meanSphere(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                Integer kernelSizeX,
                                Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image2d",
                 parameters);
  }

  @Deprecated
  public static void meanIJ(CLKernelExecutor clke,
                            ClearCLImage src,
                            ClearCLImage dst,
                            Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image2d_ij",
                 parameters);
  }

  @Deprecated
  public static void meanIJ(CLKernelExecutor clke,
                            ClearCLBuffer src,
                            ClearCLBuffer dst,
                            Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image2d_ij",
                 parameters);
  }

  public static void meanSphere(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                Integer kernelSizeX,
                                Integer kernelSizeY,
                                Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image3d",
                 parameters);
  }

  public static void meanSphere(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                Integer kernelSizeX,
                                Integer kernelSizeY,
                                Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_image3d",
                 parameters);
  }

  public static void meanBox(CLKernelExecutor clke,
                             ClearCLImage src,
                             ClearCLImage dst,
                             int radiusX,
                             int radiusY,
                             int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
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

  public static void meanBox(CLKernelExecutor clke,
                             ClearCLBuffer src,
                             ClearCLBuffer dst,
                             int radiusX,
                             int radiusY,
                             int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
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

  public static void meanSliceBySliceSphere(CLKernelExecutor clke,
                                            ClearCLImage src,
                                            ClearCLImage dst,
                                            Integer kernelSizeX,
                                            Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_slicewise_image3d",
                 parameters);
  }

  public static void meanSliceBySliceSphere(CLKernelExecutor clke,
                                            ClearCLBuffer src,
                                            ClearCLBuffer dst,
                                            Integer kernelSizeX,
                                            Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "mean_slicewise_image3d",
                 parameters);
  }

  public static void medianSphere(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_image2d",
                 parameters);
  }

  public static void medianSphere(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_image2d",
                 parameters);
  }

  public static void medianSphere(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY,
                                  Integer kernelSizeZ) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_image3d",
                 parameters);
  }

  public static void medianSphere(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst,
                                  Integer kernelSizeX,
                                  Integer kernelSizeY,
                                  Integer kernelSizeZ) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_image3d",
                 parameters);
  }

  public static void medianSliceBySliceSphere(CLKernelExecutor clke,
                                              ClearCLImage src,
                                              ClearCLImage dst,
                                              Integer kernelSizeX,
                                              Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_slicewise_image3d",
                 parameters);
  }

  public static void medianSliceBySliceSphere(CLKernelExecutor clke,
                                              ClearCLBuffer src,
                                              ClearCLBuffer dst,
                                              Integer kernelSizeX,
                                              Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_slicewise_image3d",
                 parameters);
  }

  public static void medianBox(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer kernelSizeX,
                               Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_image2d",
                 parameters);
  }

  public static void medianBox(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer kernelSizeX,
                               Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_image2d",
                 parameters);
  }

  public static void medianBox(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer kernelSizeX,
                               Integer kernelSizeY,
                               Integer kernelSizeZ) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_image3d",
                 parameters);
  }

  public static void medianBox(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer kernelSizeX,
                               Integer kernelSizeY,
                               Integer kernelSizeZ) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_image3d",
                 parameters);
  }

  public static void medianSliceBySliceBox(CLKernelExecutor clke,
                                           ClearCLImage src,
                                           ClearCLImage dst,
                                           Integer kernelSizeX,
                                           Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_slicewise_image3d",
                 parameters);
  }

  public static void medianSliceBySliceBox(CLKernelExecutor clke,
                                           ClearCLBuffer src,
                                           ClearCLBuffer dst,
                                           Integer kernelSizeX,
                                           Integer kernelSizeY) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "median_box_slicewise_image3d",
                 parameters);
  }

  public static void minimumSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image2d",
                 parameters);
  }

  public static void minimumSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image2d",
                 parameters);
  }

  public static void minimumSphere(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image3d",
                 parameters);
  }

  public static void minimumSphere(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst,
                                   Integer kernelSizeX,
                                   Integer kernelSizeY,
                                   Integer kernelSizeZ) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);
    parameters.put("Nz", kernelSizeZ);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image3d",
                 parameters);
  }

  @Deprecated
  public static void minimumIJ(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image2d_ij",
                 parameters);
  }

  @Deprecated
  public static void minimumIJ(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Integer radius) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("radius", radius);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_image2d_ij",
                 parameters);
  }

  public static void minimumBox(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "filtering.cl",
                           "min_sep_image" + src.getDimension() + "d",
                           radiusToKernelSize(radiusX),
                           radiusToKernelSize(radiusY),
                           radiusToKernelSize(radiusZ),
                           radiusX,
                           radiusY,
                           radiusZ,
                           src.getDimension());
  }

  public static void minimumBox(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst,
                                int radiusX,
                                int radiusY,
                                int radiusZ) throws CLKernelException
  {
    executeSeparableKernel(clke,
                           src,
                           dst,
                           "filtering.cl",
                           "min_sep_image" + src.getDimension() + "d",
                           radiusToKernelSize(radiusX),
                           radiusToKernelSize(radiusY),
                           radiusToKernelSize(radiusZ),
                           radiusX,
                           radiusY,
                           radiusZ,
                           src.getDimension());
  }

  public static void minimumSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLImage src,
                                               ClearCLImage dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_slicewise_image3d",
                 parameters);
  }

  public static void minimumSliceBySliceSphere(CLKernelExecutor clke,
                                               ClearCLBuffer src,
                                               ClearCLBuffer dst,
                                               Integer kernelSizeX,
                                               Integer kernelSizeY) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("Nx", kernelSizeX);
    parameters.put("Ny", kernelSizeY);

    clke.execute(OCLlib.class,
                 "kernels/filtering.cl",
                 "minimum_slicewise_image3d",
                 parameters);
  }

  public static void multiplyImages(CLKernelExecutor clke,
                                    ClearCLImage src,
                                    ClearCLImage src1,
                                    ClearCLImage dst) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void multiplyImages(CLKernelExecutor clke,
                                    ClearCLBuffer src,
                                    ClearCLBuffer src1,
                                    ClearCLBuffer dst) throws CLKernelException
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
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void multiplyImageAndCoordinate(CLKernelExecutor clke,
                                                ClearCLImage src,
                                                ClearCLImage dst,
                                                Integer dimension) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dimension", dimension);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (multiplyImageAndCoordinate)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiply_pixelwise_with_coordinate_3d",
                 parameters);
  }

  public static void multiplyImageAndCoordinate(CLKernelExecutor clke,
                                                ClearCLBuffer src,
                                                ClearCLBuffer dst,
                                                Integer dimension) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dimension", dimension);
    parameters.put("dst", dst);
    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (multiplyImageAndCoordinate)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiply_pixelwise_with_coordinate_3d",
                 parameters);
  }

  public static void multiplyImageAndScalar(CLKernelExecutor clke,
                                            ClearCLImage src,
                                            ClearCLImage dst,
                                            Float scalar) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void multiplyImageAndScalar(CLKernelExecutor clke,
                                            ClearCLBuffer src,
                                            ClearCLBuffer dst,
                                            Float scalar) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("scalar", scalar);
    parameters.put("dst", dst);

    if (!checkDimensions(src.getDimension(), dst.getDimension()))
    {
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void multiplySliceBySliceWithScalars(CLKernelExecutor clke,
                                                     ClearCLImage src,
                                                     ClearCLImage dst,
                                                     float[] scalars) throws CLKernelException
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

    Map<String, Object> map = new HashMap<>();
    map.put("src", src);
    map.put("scalars", clBuffer);
    map.put("dst", dst);
    try
    {
      clke.execute(OCLlib.class,
                   "kernels/math.cl",
                   "multiplySliceBySliceWithScalars",
                   map);
    }
    catch (CLKernelException clkExc)
    {
      throw clkExc;
    }
    finally
    {
      clBuffer.close();
    }
  }

  public static void multiplySliceBySliceWithScalars(CLKernelExecutor clke,
                                                     ClearCLBuffer src,
                                                     ClearCLBuffer dst,
                                                     float[] scalars) throws CLKernelException
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
    try
    {
      clke.execute(OCLlib.class,
                   "kernels/math.cl",
                   "multiplySliceBySliceWithScalars",
                   map);
    }
    catch (CLKernelException clkExc)
    {
      throw clkExc;
    }
    finally
    {
      clBuffer.close();
    }
  }

  public static void multiplyStackWithPlane(CLKernelExecutor clke,
                                            ClearCLImage input3d,
                                            ClearCLImage input2d,
                                            ClearCLImage output3d) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", input3d);
    parameters.put("src1", input2d);
    parameters.put("dst", output3d);
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyStackWithPlanePixelwise",
                 parameters);
  }

  public static void multiplyStackWithPlane(CLKernelExecutor clke,
                                            ClearCLBuffer input3d,
                                            ClearCLBuffer input2d,
                                            ClearCLBuffer output3d) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", input3d);
    parameters.put("src1", input2d);
    parameters.put("dst", output3d);
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyStackWithPlanePixelwise",
                 parameters);
  }

  public static void power(CLKernelExecutor clke,
                           ClearCLImage src,
                           ClearCLImage dst,
                           Float exponent) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("exponent", exponent);
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "power_" + src.getDimension() + "d",
                 parameters);
  }

  public static void power(CLKernelExecutor clke,
                           ClearCLBuffer src,
                           ClearCLBuffer dst,
                           Float exponent) throws CLKernelException
  {

    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("exponent", exponent);

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "power_" + src.getDimension() + "d",
                 parameters);
  }

  public static void radialProjection(CLKernelExecutor clke,
                                      ClearCLImage src,
                                      ClearCLImage dst,
                                      Float deltaAngle) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("deltaAngle", deltaAngle);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "radialProjection3d",
                 parameters);
  }

  public static void radialProjection(CLKernelExecutor clke,
                                      ClearCLBuffer src,
                                      ClearCLBuffer dst,
                                      Float deltaAngle) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);
    parameters.put("deltaAngle", deltaAngle);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "radialProjection3d",
                 parameters);
  }

  public static void resliceBottom(CLKernelExecutor clke,
                                   ClearCLImage src,
                                   ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_bottom_3d",
                 parameters);
  }

  public static void resliceBottom(CLKernelExecutor clke,
                                   ClearCLBuffer src,
                                   ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_bottom_3d",
                 parameters);
  }

  public static void resliceLeft(CLKernelExecutor clke,
                                 ClearCLImage src,
                                 ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_left_3d",
                 parameters);
  }

  public static void resliceLeft(CLKernelExecutor clke,
                                 ClearCLBuffer src,
                                 ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_left_3d",
                 parameters);
  }

  public static void resliceRight(CLKernelExecutor clke,
                                  ClearCLImage src,
                                  ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_right_3d",
                 parameters);
  }

  public static void resliceRight(CLKernelExecutor clke,
                                  ClearCLBuffer src,
                                  ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_right_3d",
                 parameters);
  }

  public static void resliceTop(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_top_3d",
                 parameters);
  }

  public static void resliceTop(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_top_3d",
                 parameters);
  }

  public static void rotateLeft(CLKernelExecutor clke,
                                ClearCLBuffer src,
                                ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_left_" + dst.getDimension() + "d",
                 parameters);
  }

  public static void rotateLeft(CLKernelExecutor clke,
                                ClearCLImage src,
                                ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_left_" + dst.getDimension() + "d",
                 parameters);
  }

  public static void rotateRight(CLKernelExecutor clke,
                                 ClearCLBuffer src,
                                 ClearCLBuffer dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_right_" + dst.getDimension() + "d",
                 parameters);
  }

  public static void rotateRight(CLKernelExecutor clke,
                                 ClearCLImage src,
                                 ClearCLImage dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_right_" + dst.getDimension() + "d",
                 parameters);
  }

  public static void set(CLKernelExecutor clke,
                         ClearCLImageInterface clImage,
                         Float value) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("dst", clImage);
    parameters.put("value", value);

    clke.execute(OCLlib.class,
                 "kernels/set.cl",
                 "set_" + clImage.getDimension() + "d",
                 parameters);
  }

  public static void splitStack(CLKernelExecutor clke,
                                ClearCLImage clImageIn,
                                ClearCLImage... clImagesOut) throws CLKernelException
  {
    if (clImagesOut.length > 12)
    {
      throw new IllegalArgumentException("Error: splitStack does not support more than 12 stacks.");
    }
    if (clImagesOut.length == 1)
    {
      copy(clke, clImageIn, clImagesOut[0]);
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

    clke.execute(OCLlib.class,
                 "kernels/stacksplitting.cl",
                 "split_" + clImagesOut.length + "_stacks",
                 parameters);
  }

  public static void splitStack(CLKernelExecutor clke,
                                ClearCLBuffer clImageIn,
                                ClearCLBuffer... clImagesOut) throws CLKernelException
  {
    if (clImagesOut.length > 12)
    {
      throw new IllegalArgumentException("Error: splitStack does not support more than 12 stacks.");
    }
    if (clImagesOut.length == 1)
    {
      copy(clke, clImageIn, clImagesOut[0]);
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

    clke.execute(OCLlib.class,
                 "kernels/stacksplitting.cl",
                 "split_" + clImagesOut.length + "_stacks",
                 parameters);
  }

  public static void subtractImages(CLKernelExecutor clke,
                                    ClearCLImage subtrahend,
                                    ClearCLImage minuend,
                                    ClearCLImage destination) throws CLKernelException
  {
    addImagesWeighted(clke,
                      subtrahend,
                      minuend,
                      destination,
                      1f,
                      -1f);
  }

  public static void subtractImages(CLKernelExecutor clke,
                                    ClearCLBuffer subtrahend,
                                    ClearCLBuffer minuend,
                                    ClearCLBuffer destination) throws CLKernelException
  {
    addImagesWeighted(clke,
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
            clke.execute(OCLlib.class, "kernels/projections.cl", "max_project_3d_2d", parameters);
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
        maximumGreyValue;
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
          clke.execute(OCLlib.class, "kernels/projections.cl", "max_project_3d_2d", parameters);
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
      maximumGreyValue;
  }
  
  
  public static double minimumOfAllPixels(CLKernelExecutor clke, ClearCLImage clImage) {
      ClearCLImage clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLImage(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getChannelDataType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst_min", clReducedImage);
          clke.execute(OCLlib.class, "kernels/projections.cl", "min_project_3d_2d", parameters);
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
      minimumGreyValue;
  }
  
  
  public static double minimumOfAllPixels(CLKernelExecutor clke, ClearCLBuffer clImage) {
      ClearCLBuffer clReducedImage = clImage;
      if (clImage.getDimension() == 3) {
          clReducedImage = clke.createCLBuffer(new long[]{clImage.getWidth(), clImage.getHeight()}, clImage.getNativeType());
  
          HashMap<String, Object> parameters = new HashMap<>();
          parameters.put("src", clImage);
          parameters.put("dst_min", clReducedImage);
          clke.execute(OCLlib.class, "kernels/projections.cl", "min_project_3d_2d", parameters);
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
          clke.execute(OCLlib.class, "kernels/projections.cl", "sum_project_3d_2d", parameters);
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
          clke.execute(OCLlib.class, "kernels/projections.cl", "sum_project_3d_2d", parameters);
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

  public static void sumZProjection(CLKernelExecutor clke,
                                    ClearCLImage clImage,
                                    ClearCLImage clReducedImage) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImage);
    parameters.put("dst", clReducedImage);
    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "sum_project_3d_2d",
                 parameters);
  }

  public static void sumZProjection(CLKernelExecutor clke,
                                    ClearCLBuffer clImage,
                                    ClearCLBuffer clReducedImage) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImage);
    parameters.put("dst", clReducedImage);
    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "sum_project_3d_2d",
                 parameters);
  }

  public static void tenengradWeightsSliceBySlice(CLKernelExecutor clke,
                                                  ClearCLImage clImageOut,
                                                  ClearCLImage clImageIn) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", clImageIn);
    parameters.put("dst", clImageOut);
    clke.execute(OCLlib.class,
                 "kernels/tenengradFusion.cl",
                 "tenengrad_weight_unnormalized_slice_wise",
                 parameters);
  }

  public static void tenengradFusion(CLKernelExecutor clke,
                                     ClearCLImage clImageOut,
                                     float[] blurSigmas,
                                     ClearCLImage... clImagesIn) throws CLKernelException
  {
    tenengradFusion(clke, clImageOut, blurSigmas, 1.0f, clImagesIn);
  }

  public static void tenengradFusion(CLKernelExecutor clke,
                                     ClearCLImage clImageOut,
                                     float[] blurSigmas,
                                     float exponent,
                                     ClearCLImage... clImagesIn) throws CLKernelException
  {
    if (clImagesIn.length > 12)
    {
      throw new IllegalArgumentException("Error: tenengradFusion does not support more than 12 stacks.");
    }
    if (clImagesIn.length == 1)
    {
      copy(clke, clImagesIn[0], clImageOut);
    }
    if (clImagesIn.length == 0)
    {
      throw new IllegalArgumentException("Error: tenengradFusion didn't get any output images.");
    }
    if (!clImagesIn[0].isFloat())
    {
      System.out.println("Warning: tenengradFusion may only work on float images!");
    }

    ClearCLImage temporaryImage = null;
    ClearCLImage temporaryImage2 = null;
    ClearCLImage[] temporaryImages = null;
    try
    {
      HashMap<String, Object> lFusionParameters = new HashMap<>();
      temporaryImage = clke.createCLImage(clImagesIn[0]);
      if (Math.abs(exponent - 1.0f) > 0.0001)
      {
        temporaryImage2 = clke.createCLImage(clImagesIn[0]);
      }

      temporaryImages = new ClearCLImage[clImagesIn.length];
      for (int i = 0; i < clImagesIn.length; i++)
      {
        HashMap<String, Object> parameters = new HashMap<>();
        temporaryImages[i] = clke.createCLImage(clImagesIn[i]);
        parameters.put("src", clImagesIn[i]);
        parameters.put("dst", temporaryImage);

        clke.execute(OCLlib.class,
                     "kernels/tenengradFusion.cl",
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

      clke.execute(OCLlib.class,
                   "kernels/tenengradFusion.cl",
                   String.format("tenengrad_fusion_with_provided_weights_%d_images",
                                 clImagesIn.length),
                   lFusionParameters);
    }
    catch (CLKernelException clkExc)
    {
      throw clkExc;
    }
    finally
    {
      if (temporaryImage != null)
      {
        temporaryImage.close();
      }
      if (temporaryImages != null)
      {
        for (ClearCLImage tmpImg : temporaryImages)
        {
          if (tmpImg != null)
          {

            tmpImg.close();
          }
        }
      }

      if (temporaryImage2 != null)
      {
        temporaryImage2.close();
      }

    }
  }

  public static void threshold(CLKernelExecutor clke,
                               ClearCLImage src,
                               ClearCLImage dst,
                               Float threshold) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/thresholding.cl",
                 "apply_threshold_" + src.getDimension() + "d",
                 parameters);
  }

  public static void threshold(CLKernelExecutor clke,
                               ClearCLBuffer src,
                               ClearCLBuffer dst,
                               Float threshold) throws CLKernelException
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

    clke.execute(OCLlib.class,
                 "kernels/thresholding.cl",
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
