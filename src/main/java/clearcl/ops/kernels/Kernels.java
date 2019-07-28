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
 * Re-arranged and organized for ClearCL use by Nico Stuurman (UCSF), 2019
 * 
 * For documentation, see: https://clij.github.io/clij-docs/referenceJava
 * 
 * Please copy into javadoc here whenever you have a chance!
 */
public class Kernels
{

  /**
   * Computes the absolute value of every individual pixel x in a given image.
   * f(x) = |x|
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param dst
   *          output image
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

  /**
   * Adds a scalar value s to all pixels x of a given image X.
   * 
   * f(x, s) = x + s
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param dst
   *          output image
   * @param scalar
   *          scalar value to be added to image/buffer
   * @throws CLKernelException
   */
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

  /*
    public static void affineTransform(CLKernelExecutor clke, ClearCLBuffer src, ClearCLBuffer dst, AffineTransform3D at) {
        at = at.inverse();
        float[] matrix = AffineTransform.matrixToFloatArray(at);
        return affineTransform(clke, src, dst, matrix);
    }
    */

  /**
   * Applies the given affine-transform to the input image
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param dst
   *          image resulting from the affinetransform
   * @param matrix
   *          matrix (dimensions????) describing the affine transform
   * @throws CLKernelException
   */
  public static void affineTransform(CLKernelExecutor clke,
                                     ClearCLImageInterface src,
                                     ClearCLImageInterface dst,
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
                                      ClearCLImageInterface src,
                                      ClearCLImageInterface vectorX,
                                      ClearCLImageInterface vectorY,
                                      ClearCLImageInterface dst) throws CLKernelException
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
                                      ClearCLImageInterface src,
                                      ClearCLImageInterface vectorX,
                                      ClearCLImageInterface vectorY,
                                      ClearCLImageInterface vectorZ,
                                      ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Determines the maximum projection of an image along Z. Furthermore, another
   * image is generated containing the z-index (zero based) where the maximum
   * was found.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          src image
   * @param dst_max
   *          result of max int. z projection
   * @param dst_arg
   *          image with index (zero-based) of max intensity pixel
   * @throws CLKernelException
   */
  public static void argMaximumZProjection(CLKernelExecutor clke,
                                           ClearCLImageInterface src,
                                           ClearCLImageInterface dst_max,
                                           ClearCLImageInterface dst_arg) throws CLKernelException
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

  /**
   * Computes a binary image (containing pixel values 0 and 1) from two images X
   * and Y by connecting pairs of pixels x and y with the binary AND operator &.
   * All pixel values except 0 in the input images are interpreted as 1.
   * 
   * f(x, y) = x & y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src1
   *          one input image
   * @param src2
   *          the other input image
   * @param dst
   *          binary output image
   * @throws CLKernelException
   */
  public static void binaryAnd(CLKernelExecutor clke,
                               ClearCLImageInterface src1,
                               ClearCLImageInterface src2,
                               ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes a binary image (containing pixel values 0 and 1) from two images X
   * and Y by connecting pairs of pixels x and y with the binary XOR operator.
   * The output will always be 1 if either of the inputs is 1 and will be 0 if
   * both of the inputs are 0 or 1 All pixel values except 0 in the input images
   * are interpreted as 1.
   * 
   * f(x, y) = x XOR y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src1
   *          one input image
   * @param src2
   *          the other input image
   * @param dst
   *          binary output image
   * @throws CLKernelException
   */
  public static void binaryXOr(CLKernelExecutor clke,
                               ClearCLImageInterface src1,
                               ClearCLImageInterface src2,
                               ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes a binary image (containing pixel values 0 and 1) from two images X
   * and Y by connecting pairs of pixels x and y with the binary NOT operator !.
   * All pixel values except 0 in the input images are interpreted as 1.
   * 
   * f(x, y) = x ! y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          binary output image
   * @throws CLKernelException
   */
  public static void binaryNot(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src1", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/binaryProcessing.cl",
                 "binary_not_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * Computes a binary image (containing pixel values 0 and 1) from two images X
   * and Y by connecting pairs of pixels x and y with the binary OR operator |.
   * All pixel values except 0 in the input images are interpreted as 1.
   * 
   * f(x, y) = x | y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src1
   *          one input image
   * @param src2
   *          the other input image
   * @param dst
   *          binary output image
   * @throws CLKernelException
   */
  public static void binaryOr(CLKernelExecutor clke,
                              ClearCLImageInterface src1,
                              ClearCLImageInterface src2,
                              ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes the Gaussian blurred image of an image given two sigma values in X
   * and Y. Thus, the filterkernel can have non-isotropic shape. The ‘fast’
   * implementation is done separable. In case a sigma equals zero, the
   * direction is not blurred.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          one input image
   * @param dst
   *          binary output image
   * @param blurSigmaX
   *          Sigma value in the X direction
   * @param blurSigmaY
   *          Sigma value in the Y direction
   * @throws CLKernelException
   */
  public static void blur(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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

  /**
   * Computes the Gaussian blurred image of an image given three sigma values in
   * X, Y, and Z. Thus, the filterkernel can have non-isotropic shape. The
   * ‘fast’ implementation is done separable. In case a sigma equals zero, the
   * direction is not blurred.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          one input image
   * @param dst
   *          binary output image
   * @param blurSigmaX
   *          Sigma value in the X direction
   * @param blurSigmaY
   *          Sigma value in the Y direction
   * @param blurSigmaZ
   *          Sigma value in the Z direction
   * @throws CLKernelException
   */
  public static void blur(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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
                                               ClearCLImageInterface src,
                                               ClearCLImageInterface dst,
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
                                                           ClearCLImageInterface src,
                                                           ClearCLImageInterface dst,
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
                                               ClearCLImageInterface src,
                                               ClearCLImageInterface dst,
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

  public static void blurSliceBySlice(CLKernelExecutor clke,
                                      ClearCLImageInterface src,
                                      ClearCLImageInterface dst,
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
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst) throws CLKernelException
  {
    copyInternal(clke,
                 src,
                 dst,
                 src.getDimension(),
                 dst.getDimension());
  }

  public static void copySlice(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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

  public static void crop(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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
                                      ClearCLImageInterface src1,
                                      ClearCLImageInterface meanSrc1,
                                      ClearCLImageInterface src2,
                                      ClearCLImageInterface meanSrc2,
                                      ClearCLImageInterface dst,
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

  /**
   * Detects local maxima in a given square/cubic neighborhood. Pixels in the
   * resulting image are set to 1 if there is no other pixel in a given radius
   * which has a higher intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param radius
   *          radius of square/cubic neighborhood to look for higher intensity
   *          pixels
   * @throws CLKernelException
   */
  public static void detectMaximaBox(CLKernelExecutor clke,
                                     ClearCLImageInterface src,
                                     ClearCLImageInterface dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, true);
  }

  /**
   * Detects local maxima in a given square neighborhood in a stack. The input
   * image stack is processed slice by slice. Pixels in the resulting image are
   * set to 1 if there is no other pixel in a given radius which has a higher
   * intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image stack
   * @param dst
   *          output image stack
   * @param radius
   *          radius of square neighborhood to look for higher intensity pixels
   * @throws CLKernelException
   */
  public static void detectMaximaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLImageInterface src,
                                                 ClearCLImageInterface dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, true);
  }

  /**
   * Detects local minima in a given square/cubic neighborhood. Pixels in the
   * resulting image are set to 1 if there is no other pixel in a given radius
   * which has a lower intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param radius
   *          radius of square/cubic neighborhood to look for lower intensity
   *          pixels
   * @throws CLKernelException
   */
  public static void detectMinimaBox(CLKernelExecutor clke,
                                     ClearCLImageInterface src,
                                     ClearCLImageInterface dst,
                                     Integer radius) throws CLKernelException
  {
    detectOptima(clke, src, dst, radius, false);
  }

  /**
   * Detects local minima in a given square neighborhood in a stack. The input
   * image stack is processed slice by slice. Pixels in the resulting image are
   * set to 1 if there is no other pixel in a given radius which has a lower
   * intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image stack
   * @param dst
   *          output image stack
   * @param radius
   *          radius of square neighborhood to look for higher intensity pixels
   * @throws CLKernelException
   */
  public static void detectMinimaSliceBySliceBox(CLKernelExecutor clke,
                                                 ClearCLImageInterface src,
                                                 ClearCLImageInterface dst,
                                                 Integer radius) throws CLKernelException
  {
    detectOptimaSliceBySlice(clke, src, dst, radius, false);
  }

  /**
   * Detects local minima/maxima in a given square/cubic neighborhood. Pixels in
   * the resulting image are set to 1 if there is no other pixel in a given
   * radius which has a lower/higher intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param radius
   *          radius of square/cubic neighborhood to look for lower/higher
   *          intensity pixels
   * @param detectMaxima
   *          when true, detects maxima, otherwise minima
   * @throws CLKernelException
   */
  public static void detectOptima(CLKernelExecutor clke,
                                  ClearCLImageInterface src,
                                  ClearCLImageInterface dst,
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

  /**
   * Detects local minima/maxima in a given square neighborhood in a stack. The
   * input image stack is processed slice by slice. Pixels in the resulting
   * image are set to 1 if there is no other pixel in a given radius which has a
   * lower/higher intensity, and to 0 otherwise.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image stack
   * @param dst
   *          output image stack
   * @param radius
   *          radius of square neighborhood to look for lower/higher intensity
   *          pixels
   * @param detectMaxima
   *          true: look for maxima, false: look for minima
   * @throws CLKernelException
   */
  public static void detectOptimaSliceBySlice(CLKernelExecutor clke,
                                              ClearCLImageInterface src,
                                              ClearCLImageInterface dst,
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
                                          ClearCLImageInterface src,
                                          ClearCLImageInterface dst,
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
                                                      ClearCLImageInterface src,
                                                      ClearCLImageInterface dst,
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

  /**
   * Computes a binary image with pixel values 0 and 1 containing the binary
   * dilation of a given input image. The dilation takes the Moore-neighborhood
   * (8 pixels in 2D and 26 pixels in 3d) into account. The pixels in the input
   * image with pixel value not equal to 0 will be interpreted as 1.
   * 
   * This method is comparable to the ‘Dilate’ menu in ImageJ in case it is
   * applied to a 2D image. The only difference is that the output image
   * contains values 0 and 1 instead of 0 and 255.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void dilateBox(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst) throws CLKernelException
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
                                           ClearCLImageInterface src,
                                           ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes a binary image with pixel values 0 and 1 containing the binary
   * dilation of the given input image. The dilation takes the
   * von-Neumann-neighborhood (4 pixels in 2D and 6 pixels in 3d) into account.
   * Pixels in the input image with pixel value not equal to 0 will be
   * interpreted as 1.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void dilateSphere(CLKernelExecutor clke,
                                  ClearCLImageInterface src,
                                  ClearCLImageInterface dst) throws CLKernelException
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
                                              ClearCLImageInterface src,
                                              ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Divides input images src (x) and src1 (y) by each other pixel wise.
   * 
   * f(x, y) = x / y
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param src1
   *          input image y (divisor)
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void divideImages(CLKernelExecutor clke,
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
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "dividePixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  public static void downsample(CLKernelExecutor clke,
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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
                                                      ClearCLImageInterface src,
                                                      ClearCLImageInterface dst) throws CLKernelException
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
                                 ClearCLImageInterface src,
                                 ClearCLImageInterface dst) throws CLKernelException
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
                                             ClearCLImageInterface src,
                                             ClearCLImageInterface dst) throws CLKernelException
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
                              ClearCLImageInterface src,
                              ClearCLImageInterface dst) throws CLKernelException
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
                                          ClearCLImageInterface src,
                                          ClearCLImageInterface dst) throws CLKernelException
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
   * Flips an image in X, Y, and/or Z direction depending on boolean flags.
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param flipx
   *          when true, will flip in x direction
   * @param flipy
   *          when true, will flip in y direction
   * @param flipz
   *          when true, will flip in z direction
   * @throws CLKernelException
   */
  public static void flip(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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

  /**
   * Flips an image in X,and/or Y direction depending on boolean flags.
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param flipx
   *          when true, will flip in x direction
   * @param flipy
   *          when true, will flip in y direction
   * @throws CLKernelException
   */
  public static void flip(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
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

  /**
   * Computes the gradient of gray values along X. Assume that a, b and c are
   * three adjacent pixels in X direction. The "b" pixel in the output image
   * will be assigned the value: c - a;
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          ouput image
   * @throws CLKernelException
   */
  public static void gradientX(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes the gradient of gray values along Y. Assume that a, b and c are
   * three adjacent pixels in Y direction. The "b" pixel in the output image
   * will be assigned the value: c - a;
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          ouput image
   * @throws CLKernelException
   */
  public static void gradientY(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Computes the gradient of gray values along Z. Assume that a, b and c are
   * three adjacent pixels in Z direction. The "b" pixel in the output image
   * will be assigned the value: c - a;
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          ouput image
   * @throws CLKernelException
   */
  public static void gradientZ(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Calculates a histogram from the input Buffer, and places the histogram
   * values in the dstHistogram, Create the dstHistogram as follows:
   * 
   * long[] histDims = {numberOfBins,1,1};
   * 
   * ClearCLBuffer histogram = clke.createCLBuffer(histDims,
   * NativeTypeEnum.Float);
   * 
   * @param clke
   *          - Executor that holds ClearCL context instance
   * @param src
   *          input image
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

    partialHistograms.close();
  }

  /**
   * Calculates a histogram from the input Buffer, and places the histogram
   * values in the dstHistogram. Size (number of bins) of the histogram is
   * determined by the size of dstHistogram,
   * 
   * long[] histDims = {numberOfBins,1,1};
   * 
   * ClearCLBuffer histogram = clke.createCLBuffer(histDims,
   * NativeTypeEnum.Float);
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
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

  /**
   * Computes the negative value of all pixels in a given image. It is
   * recommended to convert images to 32-bit float before applying this
   * operation.
   * 
   * f(x) = - x
   * 
   * For binary images, use binaryNot.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param input3d
   *          input image
   * @param output3d
   *          output image
   * @throws CLKernelException
   */
  public static void invert(CLKernelExecutor clke,
                            ClearCLImageInterface input3d,
                            ClearCLImageInterface output3d) throws CLKernelException
  {
    multiplyImageAndScalar(clke, input3d, output3d, -1f);
  }

  /**
   * Computes a binary image with pixel values 0 and 1 depending on if a pixel
   * value x in the input image was above or equal to the value of the
   * corresponding pixel in the mask image (m).
   * 
   * f(x) = (1 if (x >= m)); (0 otherwise)
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          ouput image
   * @param mask
   * @throws CLKernelException
   */
  public static void localThreshold(CLKernelExecutor clke,
                                    ClearCLImageInterface src,
                                    ClearCLImageInterface dst,
                                    ClearCLImageInterface mask) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();

    parameters.clear();
    parameters.put("local_threshold", mask);
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

  /**
   * Computes a masked image by applying a mask to an image. All pixel values x
   * of the input image will be copied to the output image when the
   * corresponding mask pixel is 1, otherwise the output will be set to 0.
   * 
   * f(x,m) = (x if (m != 0); (0 otherwise))
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param dst
   *          output image
   * @param mask
   *          mask image
   * @throws CLKernelException
   */
  public static void mask(CLKernelExecutor clke,
                          ClearCLImageInterface src,
                          ClearCLImageInterface dst,
                          ClearCLImageInterface mask) throws CLKernelException
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

  /**
   * Computes a masked image by applying a 2D mask to an image stack. All pixels
   * of the input image will be copied to the output image when the
   * corresponding pixel in the mask image is not zero. Otherwise the output
   * image will be set to 0.
   * 
   * f(x,m) = (x if (m != 0); (0 otherwise))
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * 
   * @param dst
   *          output image
   * @param mask
   *          mask image
   * @throws CLKernelException
   */
  public static void maskStackWithPlane(CLKernelExecutor clke,
                                        ClearCLImageInterface src,
                                        ClearCLImageInterface dst,
                                        ClearCLImageInterface mask) throws CLKernelException
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
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst,
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
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst,
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
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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
                                               ClearCLImageInterface src,
                                               ClearCLImageInterface dst,
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
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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

  /**
   * Computes the maximum of a pair of pixel values x, y from input images X and
   * Y.
   * 
   * f(x, y) = max(x, y)
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param src1
   *          input image y *
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void maximumImages(CLKernelExecutor clke,
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
      throw new IllegalArgumentException("Error: number of dimensions don't match! (maximumImages)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "maxPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * Computes the maximum of a constant scalar s pixel and the pixel values x of
   * the input image.
   * 
   * f(x, s) = max(x, s)
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param dst
   *          output image
   * @param scalarS
   *          value to compare to input image pixel values
   * @throws CLKernelException
   */
  public static void maximumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLImageInterface src,
                                           ClearCLImageInterface dst,
                                           Float scalarS) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", scalarS);
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

  /**
   * Computes the minimum of a pair of pixel values x, y from input images X and
   * Y.
   * 
   * f(x, y) = min(x, y)
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image X
   * @param src1
   *          input image Y
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void minimumImages(CLKernelExecutor clke,
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
      throw new IllegalArgumentException("Error: number of dimensions don't match! (minimumImages)");
    }
    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "minPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * Computes the minimum of a constant scalar s and pixel values x of the input
   * image.
   * 
   * f(x, s) = min(x, s)
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param dst
   *          output image
   * @param scalarS
   *          value to compare to input image pixel values
   * @throws CLKernelException
   */
  public static void minimumImageAndScalar(CLKernelExecutor clke,
                                           ClearCLImageInterface src,
                                           ClearCLImageInterface dst,
                                           Float scalarS) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("valueB", scalarS);
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

  /**
   * Assigns the maximum pixel value of the input stack to the corresponding
   * pixel of the output image .
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void maximumZProjection(CLKernelExecutor clke,
                                        ClearCLImageInterface src,
                                        ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_max", dst);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "max_project_3d_2d",
                 parameters);

  }

  /**
   * Assigns the minimum pixel value of the input stack and assigns that value
   * to the corresponding pixel of the output image .
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void minimumZProjection(CLKernelExecutor clke,
                                        ClearCLImageInterface src,
                                        ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst_min", dst);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "min_project_3d_2d",
                 parameters);
  }

  /**
   * Determines minimum and maximum of the input image
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image
   * @param nrReductions
   *          Nr. of paralel operations. 36 seems to be a nice number.
   * @return float[2], first number is the minimum, the second is the maximum
   * @throws CLKernelException
   */
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

  /**
   * Assigns the mean pixel value of the input stack to the corresponding pixel
   * of the output image .
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void meanZProjection(CLKernelExecutor clke,
                                     ClearCLImageInterface src,
                                     ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/projections.cl",
                 "mean_project_3d_2d",
                 parameters);
  }

  /**
   * Unclear what this function is doing. This is the original description:
   * 
   * Determines the maximum projection of an image along a given dimension.
   * Furthermore, the X and Y dimensions of the resulting image must be
   * specified by the user according to its definition: X = 0 Y = 1 Z = 2
   * 
   * @param clke
   * @param src
   * @param dst_max
   * @param projectedDimensionX
   * @param projectedDimensionY
   * @param projectedDimension
   * @throws CLKernelException
   */
  public static void maximumXYZProjection(CLKernelExecutor clke,
                                          ClearCLImageInterface src,
                                          ClearCLImageInterface dst_max,
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
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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
                            ClearCLImageInterface src,
                            ClearCLImageInterface dst,
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
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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
                             ClearCLImageInterface src,
                             ClearCLImageInterface dst,
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
                                            ClearCLImageInterface src,
                                            ClearCLImageInterface dst,
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
                                  ClearCLImageInterface src,
                                  ClearCLImageInterface dst,
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
                                  ClearCLImageInterface src,
                                  ClearCLImageInterface dst,
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
                                              ClearCLImageInterface src,
                                              ClearCLImageInterface dst,
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
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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
                                           ClearCLImageInterface src,
                                           ClearCLImageInterface dst,
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
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst,
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
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst,
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
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst,
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
                                               ClearCLImageInterface src,
                                               ClearCLImageInterface dst,
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

  /**
   * Multiplies all pairs of pixel values x and y from two input image X and Y.
   * 
   * f(x, y) = x * y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image x
   * @param src1
   *          input image y
   * @param dst
   *          output image
   * @throws CLKernelException
   */
  public static void multiplyImages(CLKernelExecutor clke,
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
      throw new IllegalArgumentException("Error: number of dimensions don't match! (addImageAndScalar)");
    }

    clke.execute(OCLlib.class,
                 "kernels/math.cl",
                 "multiplyPixelwise_" + src.getDimension() + "d",
                 parameters);
  }

  /**
   * TODO: what does this function do?
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image X
   * @param dst
   *          output image
   * @param dimension
   *          unclear what this is or what it is for
   * @throws CLKernelException
   */
  public static void multiplyImageAndCoordinate(CLKernelExecutor clke,
                                                ClearCLImageInterface src,
                                                ClearCLImageInterface dst,
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

  /**
   * Multiplies all pixels value x in input image X with a constant scalar s.
   * 
   * f(x, s) = x * s
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          input image X
   * @param dst
   *          output image
   * @param scalar
   *          value to multiply input with
   * @throws CLKernelException
   */
  public static void multiplyImageAndScalar(CLKernelExecutor clke,
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
                 "multiplyScalar_" + src.getDimension() + "d",
                 parameters);
  }

  public static void multiplySliceBySliceWithScalars(CLKernelExecutor clke,
                                                     ClearCLImageInterface src,
                                                     ClearCLImageInterface dst,
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

  /**
   * Multiplies all pairs of pixel values x and y from input image stack X and
   * 2D input image Y. x and y are at the same spatial position within a plane.
   * 
   * f(x, y) = x * y
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param input3d
   *          input stack X
   * @param input2d
   *          input plane Y
   * @param output3d
   *          output stack
   * @throws CLKernelException
   */
  public static void multiplyStackWithPlane(CLKernelExecutor clke,
                                            ClearCLImageInterface input3d,
                                            ClearCLImageInterface input2d,
                                            ClearCLImageInterface output3d) throws CLKernelException
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

  /**
   * Computes all pixels value x to the power of the given exponent a.
   * 
   * f(x, a) = x * a
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @param exponent
   *          Exponent to be applied to each pixel value
   * @throws CLKernelException
   */
  public static void power(CLKernelExecutor clke,
                           ClearCLImageInterface src,
                           ClearCLImageInterface dst,
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
                                      ClearCLImageInterface src,
                                      ClearCLImageInterface dst,
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

  /**
   * Flips Y and Z axis of an image stack. This operation is similar to ImageJs
   * ‘Reslice [/]’ method but does not offer certain option such as
   * interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void resliceBottom(CLKernelExecutor clke,
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst) throws CLKernelException
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

  /**
   * Flips X, Y and Z axis of an image stack. This operation is similar to
   * ImageJs ‘Reslice [/]’ method but does not offer certain option such as
   * interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void resliceLeft(CLKernelExecutor clke,
                                 ClearCLImageInterface src,
                                 ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_left_3d",
                 parameters);
  }

  /**
   * Flips X and Z axis of an image stack. This operation is similar to ImageJs
   * ‘Reslice [/]’ method but does not offer certain option such as
   * interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void resliceRight(CLKernelExecutor clke,
                                  ClearCLImageInterface src,
                                  ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_right_3d",
                 parameters);
  }

  /**
   * Flips Y and Z axis of an image stack. This operation is similar to ImageJs
   * ‘Reslice [/]’ method but does not offer certain option such as
   * interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void resliceTop(CLKernelExecutor clke,
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/reslicing.cl",
                 "reslice_top_3d",
                 parameters);
  }

  /**
   * Rotates a given input image by 90 degrees counter-clockwise. For that, X
   * and Y axis of an image stack are flipped.
   * 
   * This operation is similar to ImageJs ‘Reslice [/]’ method but does not
   * offer certain option such as interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void rotateLeft(CLKernelExecutor clke,
                                ClearCLImageInterface src,
                                ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_left_" + dst.getDimension() + "d",
                 parameters);
  }

  /**
   * Rotates a given input image by 90 degrees clockwise. For that, X and Y axis
   * of an image stack are flipped.
   * 
   * This operation is similar to ImageJs ‘Reslice [/]’ method but does not
   * offer certain option such as interpolation.
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void rotateRight(CLKernelExecutor clke,
                                 ClearCLImageInterface src,
                                 ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);

    clke.execute(OCLlib.class,
                 "kernels/rotate.cl",
                 "rotate_right_" + dst.getDimension() + "d",
                 parameters);
  }

  /**
   * Sets all pixel values x of input image X to a constant value v.
   * 
   * f(x) = v
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param clImage
   *          Input image
   * @param value
   *          Value to set each element to
   * @throws CLKernelException
   */
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
                                ClearCLImageInterface clImageIn,
                                ClearCLImageInterface... clImagesOut) throws CLKernelException
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

  /**
   * Subtracts input image Y from input image X (pixel by pixel)
   * 
   * f (x, y) = x - y;
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image X
   * @param src1
   *          Input image Y
   * @param dst
   *          Output image
   * @throws CLKernelException
   */
  public static void subtractImages(CLKernelExecutor clke,
                                    ClearCLImageInterface src,
                                    ClearCLImageInterface src1,
                                    ClearCLImageInterface dst) throws CLKernelException
  {
    addImagesWeighted(clke, src, src1, dst, 1f, -1f);
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

  /**
   * Assigns the sum of all pixels along the z axis of the input stack to the
   * output image
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image stack
   * @param dst
   *          Output image (planar)
   * @throws CLKernelException
   */
  public static void sumZProjection(CLKernelExecutor clke,
                                    ClearCLImageInterface src,
                                    ClearCLImageInterface dst) throws CLKernelException
  {
    HashMap<String, Object> parameters = new HashMap<>();
    parameters.put("src", src);
    parameters.put("dst", dst);
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

  /**
   * Computes a binary image with pixel values 0 and 1. All pixels x of the
   * input image with value larger than or equal to a given threshold t will be set
   * to 1 in the output image.
   * 
   * f(x,t) = (1 if (x >= t); (0 otherwise))
   * 
   * @param clke
   *          Executor that holds ClearCL context instance
   * @param src
   *          Input image
   * @param dst
   *          Output image
   * @param threshold
   *          Threshold value to apply
   * @throws CLKernelException
   */
  public static void threshold(CLKernelExecutor clke,
                               ClearCLImageInterface src,
                               ClearCLImageInterface dst,
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

  /////// private functions (utilties) //////

  private static void executeSeparableKernel(CLKernelExecutor clke,
                                             ClearCLImageInterface src,
                                             ClearCLImageInterface dst,
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

    ClearCLImageInterface temp;
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

  private static void copyInternal(CLKernelExecutor clke,
                                   ClearCLImageInterface src,
                                   ClearCLImageInterface dst,
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
