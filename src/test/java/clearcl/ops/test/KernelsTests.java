package clearcl.ops.test;

import java.io.IOException;

import clearcl.ClearCL;
import clearcl.ClearCLBuffer;
import clearcl.ClearCLContext;
import clearcl.ClearCLDevice;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.enums.HostAccessType;
import clearcl.enums.KernelAccessType;
import clearcl.enums.MemAllocMode;
import clearcl.ops.kernels.CLKernelException;
import clearcl.ops.kernels.CLKernelExecutor;
import clearcl.ops.kernels.Kernels;
import coremem.enums.NativeTypeEnum;

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
  private ClearCLContext lCLContext;
  private CLKernelExecutor lCLKE;
  final long xSize = 1024;
  final long ySize = 1024;
  long[] dimensions2D =
  { xSize, ySize };

  @Before
  public void initKernelTests() throws IOException
  {
    ClearCLBackendInterface lClearCLBackend =
                                            ClearCLBackends.getBestBackend();

    ClearCL lClearCL = new ClearCL(lClearCLBackend);

    ClearCLDevice lBestGPUDevice = lClearCL.getBestGPUDevice();

    lCLContext = lBestGPUDevice.createContext();

    lCLKE =
          new CLKernelExecutor(lCLContext,
                               clearcl.ocllib.OCLlib.class,
                               "kernels/blur.cl",
                               "gaussian_blur_image2d",
                               dimensions2D);
  }

  @After
  public void cleanupKernelTests() throws IOException
  {

    lCLKE.close();

    lCLContext.close();
  }

  @Test
  public void testBlurImage() throws IOException
  {

    ClearCLBuffer lCLsrcBuffer =
                               lCLContext.createBuffer(MemAllocMode.Best,
                                                       HostAccessType.ReadWrite,
                                                       KernelAccessType.ReadWrite,
                                                       1,
                                                       NativeTypeEnum.UnsignedShort,
                                                       dimensions2D);

    ClearCLBuffer lCldstBuffer = lCLKE.createCLBuffer(lCLsrcBuffer);

    try
    {
      Kernels.blur(lCLKE, lCLsrcBuffer, lCldstBuffer, 4.0f, 4.0f);
    }
    catch (CLKernelException clkExc)
    {
      Assert.fail(clkExc.getMessage());
    }

  }

}