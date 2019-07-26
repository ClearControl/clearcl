package clearcl.test;

import static java.lang.Thread.sleep;
import static org.junit.Assert.assertTrue;

import clearcl.*;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.enums.BuildStatus;
import clearcl.enums.ImageChannelDataType;
import coremem.enums.NativeTypeEnum;
import coremem.rgc.RessourceCleaner;

import org.junit.Test;

/**
 * Basic ressource gargabe collection (RGC) for images and buffers tests .
 *
 * @author royer
 */
public class ClearCLRGCTests
{

  /**
   * test with best backend
   *
   * @throws Exception
   *           NA
   */
  @Test
  public void testRGC() throws Exception
  {
    final ClearCLBackendInterface lClearCLBackendInterface =
                                                           ClearCLBackends.getBestBackend();

    testWithBackend(lClearCLBackendInterface);
  }

  private void testWithBackend(final ClearCLBackendInterface pClearCLBackendInterface) throws Exception
  {
    // Forces the loading of the Ressource Cleaner...
    RessourceCleaner.cleanNow();

    try (ClearCL lClearCL = new ClearCL(pClearCLBackendInterface))
    {

      final ClearCLDevice lDevice = lClearCL.getBestGPUDevice();

      ClearCLContext lMainContext = lDevice.createContext();

      final int lInitialNumberOfRegisteredObjects =
                                                  RessourceCleaner.getNumberOfRegisteredObjects();

      /*System.out.println("init num registered objects: "
                         + lInitialNumberOfRegisteredObjects);/**/

      for (int i = 0; i < 64; i++)
      {
        System.out.println("iteration: " + i);
        /*System.out.println("registered objects: "
                           + RessourceCleaner.getNumberOfRegisteredObjects());/**/

        // we intentionally allocate an image and a buffer and forget its
        // reference...
        // with the hope that the rgc machinery will free the image at GC time.
        allocate(lMainContext, lDevice, i);
        sleep(1);
        System.gc();
      }

      for (int i = 0; i < 10000; i++)
      {
        System.gc();
        sleep(1);

        if (i % 100 == 0)
          System.out.println(RessourceCleaner.getNumberOfRegisteredObjects()
                             - lInitialNumberOfRegisteredObjects);/**/
        if (RessourceCleaner.getNumberOfRegisteredObjects() <= lInitialNumberOfRegisteredObjects)
          break;
      }

      int lNumberOfRegisteredObjects =
                                     RessourceCleaner.getNumberOfRegisteredObjects();
      assertTrue(lNumberOfRegisteredObjects <= lInitialNumberOfRegisteredObjects);

    }
  }

  private void allocate(ClearCLContext pMainContext,
                        ClearCLDevice pDevice,
                        int pI)
  {
    try
    {
      ClearCLContext lLocalContext = pDevice.createContext();

      final ClearCLImage lImage =
                                pMainContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                                      1000,
                                                                      1000);

      lImage.fill(1.3f, true, false);

      final ClearCLBuffer lBuffer =
                                  pMainContext.createBuffer(NativeTypeEnum.Float,
                                                            1000 * 1000);

      lBuffer.fill((byte) 13, true);

      ClearCLProgram lProgram =
                              lLocalContext.createProgram(this.getClass(),
                                                          "test.cl");
      lProgram.addDefine("CONSTANT", "10");
      lProgram.addBuildOptionAllMathOpt();

      BuildStatus lBuildStatus = lProgram.buildAndLog();
      ClearCLKernel lKernel = lProgram.createKernel("buffersum");

      // What if we close the image or buffer manually? what would happen? bad
      // things I
      // can tell you... Unless! we do the right thing and 'neutralise' the
      // cleaner as we
      // close the image...

      if (pI % 2 == 0)
        lLocalContext.close();

      if (pI % 8 == 0)
        lKernel.close();

      if (pI % 16 == 0)
        lBuffer.close();

      if (pI % 32 == 0)
        lImage.close();

    }
    catch (Throwable pE)
    {
      pE.printStackTrace();
    }

    try
    {
      sleep(1);
    }
    catch (InterruptedException e)
    {
      e.printStackTrace();
    }
  }

}
