package clearcl.ops.test;

import static org.junit.Assert.assertEquals;

import java.io.IOException;

import clearcl.ClearCL;
import clearcl.ClearCLBuffer;
import clearcl.ClearCLContext;
import clearcl.ClearCLDevice;
import clearcl.ClearCLImage;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.KernelAccessType;
import clearcl.enums.MemAllocMode;
import clearcl.ops.kernels.CLKernelException;
import clearcl.ops.kernels.CLKernelExecutor;
import clearcl.ops.kernels.Kernels;
import coremem.enums.NativeTypeEnum;
import coremem.offheap.OffHeapMemory;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 *
 * @author nico
 */
public class KernelsTests
{
  private ClearCLContext gCLContext;
  private CLKernelExecutor gCLKE;
  final long xSize = 1024;
  final long ySize = 1024;
  final long zSize = 4;
  final long[] dimensions1D =
  { xSize * ySize };
  final long[] dimensions2D =
  { xSize, ySize };
  final long[] dimensions3D =
  { xSize, ySize, zSize };
  final long[][] allDimensions =
  { dimensions1D, dimensions2D, dimensions3D };
  ClearCLImage srcFloat, srcUByte, srcUShort;
  ClearCLImage dstFloat, dstUByte, dstUShort;
  ClearCLBuffer srcBufFloat, srcBufUByte, srcBufUShort;
  ClearCLBuffer dstBufFloat, dstBufUByte, dstBufUShort;
  ClearCLImage[] srcImages;
  ClearCLImage[] dstImages;
  ClearCLBuffer[] srcBuffers =
  { srcBufFloat, srcBufUByte, srcBufUShort };
  ClearCLBuffer[] dstBuffers =
  { dstBufFloat, dstBufUByte, dstBufUShort };

  @Before
  public void initKernelTests() throws IOException
  {
    ClearCLBackendInterface lClearCLBackend =
                                            ClearCLBackends.getBestBackend();

    ClearCL lClearCL = new ClearCL(lClearCLBackend);

    ClearCLDevice lBestGPUDevice = lClearCL.getBestGPUDevice();

    gCLContext = lBestGPUDevice.createContext();

    gCLKE = new CLKernelExecutor(gCLContext,
                                 clearcl.ocllib.OCLlib.class);

    // create src and dst images and buffers for all types to speed up testing
    // and reduce testing code

    srcFloat = gCLKE.createCLImage(dimensions2D,
                                   ImageChannelDataType.Float);
    dstFloat = gCLKE.createCLImage(srcFloat);
    srcUByte = gCLKE.createCLImage(dimensions2D,
                                   ImageChannelDataType.UnsignedInt8);
    dstUByte = gCLKE.createCLImage(srcUByte);
    srcUShort =
              gCLKE.createCLImage(dimensions2D,
                                  ImageChannelDataType.UnsignedInt16);
    dstUShort = gCLKE.createCLImage(srcUShort);

    srcBufFloat = gCLKE.createCLBuffer(dimensions2D,
                                       NativeTypeEnum.Float);
    dstBufFloat = gCLKE.createCLBuffer(srcBufFloat);
    srcBufUByte = gCLKE.createCLBuffer(dimensions2D,
                                       NativeTypeEnum.UnsignedByte);
    dstBufUByte = gCLKE.createCLBuffer(srcBufUByte);
    srcBufUShort = gCLKE.createCLBuffer(dimensions2D,
                                        NativeTypeEnum.UnsignedShort);
    dstBufUShort = gCLKE.createCLBuffer(srcBufUShort);

    srcImages = new ClearCLImage[]
    { srcFloat, srcUByte, srcUShort };
    dstImages = new ClearCLImage[]
    { dstFloat, dstUByte, dstUShort };
    srcBuffers = new ClearCLBuffer[]
    { srcBufFloat, srcBufUByte, srcBufUShort };
    dstBuffers = new ClearCLBuffer[]
    { dstBufFloat, dstBufUByte, dstBufUShort };

  }

  @After
  public void cleanupKernelTests() throws IOException
  {

    gCLKE.close();

    gCLContext.close();
  }

  @Test
  public void testAbsolute() throws IOException
  {
    // Todo: check unsigned integer types?
    try
    {
      Kernels.set(gCLKE, srcFloat, -3.0f);
      Kernels.absolute(gCLKE, srcFloat, dstFloat);
      float[] minMax = Kernels.minMax(gCLKE, dstFloat, 36);
      Assert.assertEquals(3.0f, minMax[0], 0.000001);
      Kernels.set(gCLKE, srcBufFloat, -5.0f);
      Kernels.absolute(gCLKE, srcBufFloat, dstBufFloat);
      minMax = Kernels.minMax(gCLKE, dstBufFloat, 36);
      Assert.assertEquals(5.0f, minMax[0], 0.000001);
    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }
  }

  @Test
  public void testAddImages() throws IOException
  {
    try
    {
      for (int i = 0; i < srcImages.length; i++)
      {
        ClearCLImage src2 = gCLKE.createCLImage(srcImages[i]);
        Kernels.set(gCLKE, srcImages[i], 1.0f);
        Kernels.set(gCLKE, src2, 2.0f);
        Kernels.addImages(gCLKE, srcImages[i], src2, dstImages[i]);
        float minMax[] = Kernels.minMax(gCLKE, dstImages[i], 36);
        Assert.assertEquals(3.0f, minMax[0], 0.000001);
      }
      for (int i = 0; i < srcBuffers.length; i++)
      {
        ClearCLBuffer src2 = gCLKE.createCLBuffer(srcBuffers[i]);
        Kernels.set(gCLKE, srcBuffers[i], 1.0f);
        Kernels.set(gCLKE, src2, 2.0f);
        Kernels.addImages(gCLKE, srcBuffers[i], src2, dstBuffers[i]);
        float minMax[] = Kernels.minMax(gCLKE, dstBuffers[i], 36);
        Assert.assertEquals(3.0f, minMax[0], 0.000001);
      }
    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

  }

  @Test
  public void testBlurImage() throws IOException
  {

    ClearCLBuffer lCLsrcBuffer =
                               gCLContext.createBuffer(MemAllocMode.Best,
                                                       HostAccessType.ReadWrite,
                                                       KernelAccessType.ReadWrite,
                                                       1,
                                                       NativeTypeEnum.UnsignedShort,
                                                       dimensions2D);

    ClearCLBuffer lCldstBuffer = gCLKE.createCLBuffer(lCLsrcBuffer);

    try
    {
      Kernels.blur(gCLKE, lCLsrcBuffer, lCldstBuffer, 4.0f, 4.0f);
    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }
  }

  @Test
  public void testMinMaxBuffer()
  {
    for (long[] lDimensions : allDimensions)
    {
      ClearCLBuffer lCLBuffer =
                              gCLContext.createBuffer(MemAllocMode.Best,
                                                      HostAccessType.ReadWrite,
                                                      KernelAccessType.ReadWrite,
                                                      1,
                                                      NativeTypeEnum.Float,
                                                      lDimensions);
      OffHeapMemory lBuffer =
                            OffHeapMemory.allocateFloats(lCLBuffer.getLength());

      float lJavaMin = Float.POSITIVE_INFINITY;
      float lJavaMax = Float.NEGATIVE_INFINITY;
      for (int i = 0; i < lCLBuffer.getLength(); i++)
      {
        float lValue = 1f / (1f + i);
        lJavaMin = Math.min(lJavaMin, lValue);
        lJavaMax = Math.max(lJavaMax, lValue);
        lBuffer.setFloatAligned(i, lValue);
      }

      // System.out.println("lJavaMin=" + lJavaMin);
      // System.out.println("lJavaMax=" + lJavaMax);

      lCLBuffer.readFrom(lBuffer, true);
      try
      {
        float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLBuffer, 128);
        assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
        assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

      }
      catch (CLKernelException clkExc)
      {
        Assert.fail(clkExc.getMessage());
      }

      lCLBuffer.close();
    }

  }

  @Test
  public void testMinMaxImageFloat()
  {

    ClearCLImage lCLImage =
                          gCLKE.createCLImage(dimensions2D,
                                              ImageChannelDataType.Float);

    long size = lCLImage.getWidth() * lCLImage.getHeight();
    OffHeapMemory lBuffer = OffHeapMemory.allocateFloats(size);

    float lJavaMin = Float.POSITIVE_INFINITY;
    float lJavaMax = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++)
    {
      float lValue = 1f / (1f + i);
      lJavaMin = Math.min(lJavaMin, lValue);
      lJavaMax = Math.max(lJavaMax, lValue);
      lBuffer.setFloatAligned(i, lValue);
    }

    lCLImage.readFrom(lBuffer, true);
    try
    {
      float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
      assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
      assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

    lCLImage.close();
  }

  @Test
  public void testMinMaxImageUI16()
  {

    ClearCLImage lCLImage =
                          gCLKE.createCLImage(dimensions2D,
                                              ImageChannelDataType.UnsignedInt16);

    long size = lCLImage.getWidth() * lCLImage.getHeight();
    OffHeapMemory lBuffer = OffHeapMemory.allocateShorts(size);

    float lJavaMin = Float.POSITIVE_INFINITY;
    float lJavaMax = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++)
    {
      float lValue = 23000f / (1f + i) + 129.0f;
      lJavaMin = (int) Math.min(lJavaMin, lValue);
      lJavaMax = (int) Math.max(lJavaMax, lValue);
      short sv = (short) (0xFFFF & (int) lValue);
      lBuffer.setShortAligned(i, sv);
    }

    lCLImage.readFrom(lBuffer, true);
    try
    {
      float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
      assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
      assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

    lCLImage.close();
  }

  @Test
  public void testMinMaxImageUI8()
  {

    ClearCLImage lCLImage =
                          gCLKE.createCLImage(dimensions2D,
                                              ImageChannelDataType.UnsignedInt8);

    long size = lCLImage.getWidth() * lCLImage.getHeight();
    OffHeapMemory lBuffer = OffHeapMemory.allocateBytes(size);

    float lJavaMin = Float.POSITIVE_INFINITY;
    float lJavaMax = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++)
    {
      float lValue = 220.0f / (1f + i) + 5.0f;
      lJavaMin = (int) Math.min(lJavaMin, lValue);
      lJavaMax = (int) Math.max(lJavaMax, lValue);
      byte sv = (byte) (0xFF & ((int) lValue));
      lBuffer.setByteAligned(i, sv);
    }

    lCLImage.readFrom(lBuffer, true);
    try
    {
      float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
      assertEquals(lJavaMin, lOpenCLMinMax[0], 0.0001);
      assertEquals(lJavaMax, lOpenCLMinMax[1], 0.0001);

    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

    lCLImage.close();
  }

  @Test
  public void testMinimumImages()
  {
    ImageChannelDataType[] types =
    { ImageChannelDataType.Float,
      ImageChannelDataType.UnsignedInt16,
      ImageChannelDataType.UnsignedInt8 };
    for (ImageChannelDataType type : types)
    {
      testMinimumImages(type);
    }
  }

  public void testMinimumImages(ImageChannelDataType type)
  {
    ClearCLImage src1 = gCLKE.createCLImage(dimensions2D, type);
    ClearCLImage src2 = gCLKE.createCLImage(src1);
    ClearCLImage dst = gCLKE.createCLImage(src1);

    try
    {
      Kernels.set(gCLKE, src1, 3.0f);
      Kernels.set(gCLKE, src2, 1.0f);
      Kernels.minimumImages(gCLKE, src1, src2, dst);
      // TODO: test that src2 and dst are identical

    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }
  }

  @Test
  public void testHistogram()
  {
    ClearCLImage lCLImage =
                          gCLKE.createCLImage(dimensions2D,
                                              ImageChannelDataType.UnsignedInt16);

    long size = lCLImage.getWidth() * lCLImage.getHeight();
    OffHeapMemory lBuffer = OffHeapMemory.allocateShorts(size);

    float lJavaMin = Float.POSITIVE_INFINITY;
    float lJavaMax = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++)
    {
      float lValue = 23000f / (1f + i) + 129.0f;
      lJavaMin = (int) Math.min(lJavaMin, lValue);
      lJavaMax = (int) Math.max(lJavaMax, lValue);
      short sv = (short) (0xFFFF & (int) lValue);
      lBuffer.setShortAligned(i, sv);
    }

    lCLImage.readFrom(lBuffer, true);
    try
    {
      // OpenCL uses uShort to index the histogram
      // however, other limits - possibly hardware related - are reached
      // at smaller size already. For now, just be sure that sizes other
      // than 256 actually work
      int[] histLengths =
      { 256, 2048 };
      for (int histLength : histLengths)
      {
        // CPU histogram calculation
        int[] cpuHist = new int[histLength];
        int min = (int) lJavaMin;
        int max = (int) lJavaMax;
        int range = max - min;
        int maxIndex = histLength - 1;
        float histLengthDivRange = (float) histLength / (float) range;

        for (long i = 0; i < size; i++)
        {
          short val = lBuffer.getShortAligned(i);
          int iVal = 0xFFFF & val;
          int index = (int) ((iVal - min) * histLengthDivRange);
          index = index > maxIndex ? maxIndex : index;
          cpuHist[index]++;
        }
        // GPU histogram calculation
        float[] lOpenCLMinMax = Kernels.minMax(gCLKE, lCLImage, 128);
        int[] gpuHist = new int[histLength];
        Kernels.histogram(gCLKE,
                          lCLImage,
                          gpuHist,
                          lOpenCLMinMax[0],
                          lOpenCLMinMax[1]);
        long sum = 0;
        for (int i = 0; i < gpuHist.length; i++)
        {
          sum += gpuHist[i];
          // System.out.println(" " + i + ": " + " CPU: " + cpuHist[i] + ", GPU:
          // " + gpuHist[i]);
          assertEquals(gpuHist[i], cpuHist[i]);
        }
        assertEquals(size, sum);
      }
    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

    lCLImage.close();
  }

}