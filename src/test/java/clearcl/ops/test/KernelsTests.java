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

import org.junit.Assert;
import org.junit.Test;

/**
 *
 * @author nico
 */
public class KernelsTests
{

  @Test
  public void testBlurImage() throws IOException
  {
    final long xSize = 1024;
    final long ySize = 1024;
    long[] dimensions =
    { xSize, ySize };

    ClearCLBackendInterface lClearCLBackend =
                                            ClearCLBackends.getBestBackend();

    try (ClearCL lClearCL = new ClearCL(lClearCLBackend))
    {
      ClearCLDevice lBestGPUDevice = lClearCL.getBestGPUDevice();

      ClearCLContext lCLContext = lBestGPUDevice.createContext();

      ClearCLBuffer lCLsrcBuffer =
                                 lCLContext.createBuffer(MemAllocMode.Best,
                                                         HostAccessType.ReadWrite,
                                                         KernelAccessType.ReadWrite,
                                                         1,
                                                         NativeTypeEnum.UnsignedShort,
                                                         dimensions);

      CLKernelExecutor lCLKE = new CLKernelExecutor(lCLContext,
                                                    clearcl.ocllib.OCLlib.class,
                                                    "kernels/blur.cl",
                                                    "gaussian_blur_image2d",
                                                    dimensions);

      ClearCLBuffer lCldstBuffer = lCLKE.createCLBuffer(lCLsrcBuffer);

      try
      {
        Kernels.blur(lCLKE, lCLsrcBuffer, lCldstBuffer, 4.0f, 4.0f);
      }
      catch (CLKernelException clkExc)
      {
        Assert.fail(clkExc.getMessage());
      }

      lCLKE.close();

      lCLContext.close();

    }
  }
}
