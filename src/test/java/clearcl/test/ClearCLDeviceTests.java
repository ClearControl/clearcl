package clearcl.test;

import static org.junit.Assert.assertTrue;

import clearcl.ClearCL;
import clearcl.ClearCLDevice;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.ClearCLBackends;
import clearcl.selector.BadDeviceSelector;
import clearcl.selector.DeviceTypeSelector;
import clearcl.selector.GlobalMemorySelector;

import org.junit.Test;

/**
 * Basic Kernel tests
 *
 * @author royer
 */
public class ClearCLDeviceTests
{

  private void testDeviceVersion_Impl(ClearCLBackendInterface pClearCLBackendInterface) throws Exception
  {
    ClearCL lClearCL = new ClearCL(pClearCLBackendInterface);

    ClearCLDevice lClearClDevice =
                                 lClearCL.getBestDevice(DeviceTypeSelector.GPU,
                                                        BadDeviceSelector.NotIntegratedIntel,
                                                        GlobalMemorySelector.MAX);

    final double version = lClearClDevice.getVersion();
    assertTrue(version > 0.);

  }

  /**
   * Test with best backend
   *
   * @throws Exception
   *           NA
   */
  @Test
  public void testDeviceVersion() throws Exception
  {
    ClearCLBackendInterface lClearCLBackendInterface =
                                                     ClearCLBackends.getBestBackend();

    testDeviceVersion_Impl(lClearCLBackendInterface);
  }

}
