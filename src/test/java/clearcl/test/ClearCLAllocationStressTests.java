package clearcl.test;

import static junit.framework.Assert.fail;

import java.io.IOException;

import clearcl.*;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.enums.BuildStatus;
import clearcl.enums.ImageChannelDataType;
import coremem.enums.NativeTypeEnum;
import coremem.rgc.Cleanable;
import coremem.rgc.Cleaner;
import coremem.rgc.RessourceCleaner;

import org.junit.Test;

/**
 * Basic ressource gargabe collection (RGC) for images and buffers tests .
 *
 * @author royer
 */
public class ClearCLAllocationStressTests
{

  public static final void freeRessource(long pResourceId)
  {
    // System.out.println("Releasing: Control(dummy) " + pResourceId);
  }

  private static class ClassWithRessource implements Cleanable
  {
    long mSomeRessource = (long) (1000 * Math.random());

    {
      RessourceCleaner.register(this);
    }

    static class MyCleaner implements Cleaner
    {
      private long mSomeRessource2;

      public MyCleaner(long pSomeRessource)
      {
        mSomeRessource2 = pSomeRessource;
      }

      @Override
      public void run()
      {
        freeRessource(mSomeRessource2);
      }
    }

    @Override
    public Cleaner getCleaner()
    {
      return new MyCleaner(mSomeRessource);
    }

  }

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
    // ClearCL.sDebugRGC = true;
    try
    {

      try (ClearCL lClearCL = new ClearCL(pClearCLBackendInterface))
      {

        for (ClearCLDevice lDevice : lClearCL.getAllDevices())
        {
          System.out.println(lDevice.getName());
          ClearCLContext lMainContext = lDevice.createContext();

          for (int i = 0; i < 100; i++)
          {
            // System.out.println(
            // "_____________________________________________________________");
            System.out.println("stress test iteration: " + i);
            /*System.out.println("Number of live contexts: "
                             + lDevice.getNumberOfLiveContexts());
            System.out.println("Number of available contexts: "
                             + lDevice.getNumberOfAvailableContexts());/**/

            allocateContexts(lDevice, lMainContext, i);
            allocateQueues(lDevice, lMainContext, i);
            allocateImages(lDevice, lMainContext, i);
            allocateBuffers(lDevice, lMainContext, i);
            allocateProgramsAndKernels(lDevice, lMainContext, i);

            // for more in depth testing:
            // allocateAllExtreme(lDevice, lMainContext, i);

            if (i % 5 == 0)
              System.gc();

            System.out.println("Number of objects in RC: " + RessourceCleaner.getNumberOfRegisteredObjects());
          }
        }

      }

    }
    catch (Throwable pE)
    {
      pE.printStackTrace();
      fail();
    }
  }

  private void allocateContexts(ClearCLDevice pDevice,
                                ClearCLContext pMainContext,
                                int pI)
  {
    for (int i = 0; i < 10; i++)
    {
      ClearCLContext lAnotherContext = pDevice.createContext();
      ClearCLImage lSomeImage =
                              lAnotherContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                                       1000,
                                                                       1000);
      if (i % 10 != 0)
        lAnotherContext.close();
    }
  }

  private void allocateQueues(ClearCLDevice pDevice,
                              ClearCLContext pMainContext,
                              int pI)
  {
    for (int i = 0; i < 10; i++)
    {
      final ClearCLQueue lLocalContextQueue =
                                            pMainContext.createQueue();
      if (i % 10 == 0)
        lLocalContextQueue.close();
    }
  }

  private void allocateImages(ClearCLDevice pDevice,
                              ClearCLContext pMainContext,
                              int pI)
  {
    for (int i = 0; i < 10; i++)
    {
      ClearCLImage lImage =
                          pMainContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                                1000,
                                                                1000);
      if (i % 10 == 0)
        lImage.close();
    }
  }

  private void allocateBuffers(ClearCLDevice pDevice,
                               ClearCLContext pMainContext,
                               int pI)
  {
    for (int i = 0; i < 10; i++)
    {
      ClearCLBuffer lBuffer =
                            pMainContext.createBuffer(NativeTypeEnum.Float,
                                                      1000 * 1000);
      if (i % 10 == 0)
        lBuffer.close();
    }
  }

  private void allocateProgramsAndKernels(ClearCLDevice pDevice,
                                          ClearCLContext pMainContext,
                                          int pI)
  {
    try
    {
      for (int i = 0; i < 10; i++)
      {
        // Program:

        ClearCLProgram lProgram = null;

        lProgram = pMainContext.createProgram(this.getClass(),
                                              "test.cl");

        lProgram.addDefine("CONSTANT", "10");
        BuildStatus lBuildStatus = lProgram.buildAndLog();
        lProgram.addBuildOptionAllMathOpt();
        lBuildStatus = lProgram.buildAndLog();

        // Kernel:
        ClearCLKernel lKernel = lProgram.createKernel("buffersum");

      }
    }
    catch (IOException pE)
    {
      pE.printStackTrace();
      fail();
    }
  }

  // Extreme stress test, only use when something is wrong, runs very slowly
  private void allocateAllExtreme(ClearCLDevice pDevice,
                                  ClearCLContext pMainContext,
                                  int pI)
  {
    try
    {
      System.out.println("LOOP BEGIN");

      // Create many OpenCL objects, sometimes we close them, we always forget
      // them,
      // and we see if anything breaks...

      // Control: dummy object that has nothing to do with ClearCL:
      ClassWithRessource lClassWithRessource =
                                             new ClassWithRessource();
      lClassWithRessource.toString();

      // Context:
      ClearCLContext lContext;
      // lContext= pDevice.createContext();
      lContext = pMainContext;

      // Another context and stuff happens...
      for (int i = 0; i < 20; i++)
      {
        ClearCLContext lAnotherContext = pDevice.createContext();
        ClearCLImage lSomeImage =
                                lAnotherContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                                         1000,
                                                                         1000);
        lSomeImage.fill(1.3f, true, false);
        if (Math.random() < 0.9)
          lAnotherContext.close();
      }

      // Queue:
      final ClearCLQueue lLocalContextQueue = lContext.createQueue();
      lLocalContextQueue.waitToFinish();
      lLocalContextQueue.waitToFinish();
      lLocalContextQueue.waitToFinish();
      lLocalContextQueue.waitToFinish();
      lLocalContextQueue.waitToFinish();
      if (Math.random() < 0.1)
        lLocalContextQueue.close();

      // Image:
      ClearCLImage lImage =
                          pMainContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                                1000,
                                                                1000);
      lImage.fill(1.3f, true, false);
      if (Math.random() < 0.1)
        lImage.close();

      // Buffer:
      ClearCLBuffer lBuffer =
                            pMainContext.createBuffer(NativeTypeEnum.Float,
                                                      1000 * 1000);
      lBuffer.fill((byte) 13, true);
      if (Math.random() < 0.1)
        lBuffer.close();

      // Program:
      ClearCLProgram lProgram =
                              lContext.createProgram(this.getClass(),
                                                     "test.cl");
      lProgram.addDefine("CONSTANT", "10");
      BuildStatus lBuildStatus = lProgram.buildAndLog();
      lProgram.addBuildOptionAllMathOpt();
      lBuildStatus = lProgram.buildAndLog();

      // Kernel:
      ClearCLKernel lKernel = lProgram.createKernel("buffersum");

      // Sometimes run the kernel:
      if (Math.random() < 0.3)
      {
        ClearCLBuffer a =
                        pMainContext.createBuffer(NativeTypeEnum.Float,
                                                  1000 * 1000);
        ClearCLBuffer b =
                        pMainContext.createBuffer(NativeTypeEnum.Float,
                                                  1000 * 1000);
        ClearCLBuffer c =
                        pMainContext.createBuffer(NativeTypeEnum.Float,
                                                  1000 * 1000);
        lKernel.setArgument("a", a);
        lKernel.setArgument("b", b);
        lKernel.setArgument("c", c);
        lKernel.setGlobalSizes(1000 * 1000);
        lKernel.run();
      }

      // Sometimes delete the kernel after use:
      if (Math.random() < 0.1)
        lKernel.close();

      // lots of images and buffers:
      for (int i = 0; i < 10; i++)
      {
        lImage =
               pMainContext.createSingleChannelImage(ImageChannelDataType.Float,
                                                     1000,
                                                     1000);

        lImage.fill(1.3f, true, false);

        lBuffer = pMainContext.createBuffer(NativeTypeEnum.Float,
                                            1000 * 1000);

        lBuffer.fill((byte) 13, true);

        // Sometimes we clean after our mess, sometimes, not.
        if (Math.random() < 0.5)
        {
          lImage.close();
          lBuffer.close();
        }
      }

      /**/

      // lContext.close();

      // lLocalContext.toString();

      System.out.println("LOOP END");

    }
    catch (Throwable pE)
    {
      pE.printStackTrace();
    }
  }

}
