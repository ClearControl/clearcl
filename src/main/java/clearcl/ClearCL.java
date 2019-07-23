package clearcl;

import java.util.ArrayList;
import java.util.HashSet;

import clearcl.abs.ClearCLBase;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.selector.BadDeviceSelector;
import clearcl.selector.DeviceName;
import clearcl.selector.DeviceSelector;
import clearcl.selector.DeviceTypeSelector;
import clearcl.selector.FastestDeviceSelector;
import clearcl.selector.GlobalMemorySelector;
import coremem.rgc.RessourceCleaner;

/**
 * ClearCL is the starting point for creating ClearCL objects for OpenCL.
 * Instances of this class are constructed by providing a backend.
 * 
 * @author royer
 */
public class ClearCL extends ClearCLBase
{

  static
  {
    // Forces the resource garbage collector to start
    RessourceCleaner.cleanNow();
  }

  // This switch should never be touched, true -> RGC on, false--> RGC off
  public static boolean sRGC = true;

  // Turns on RGC debugging.
  public static boolean sDebugRGC = false;

  HashSet<ClearCLDevice> mAccessedDeviceList = new HashSet<>();

  /**
   * Creates a ClearCL instance for a given ClearCL backend.
   * 
   * @param pClearCLBackendInterface
   *          backend
   */
  public ClearCL(ClearCLBackendInterface pClearCLBackendInterface)
  {
    super(pClearCLBackendInterface, null);
  }

  /**
   * Returns the number of available platforms
   * 
   * @return number of platforms
   */
  public int getNumberOfPlatforms()
  {
    return getBackend().getNumberOfPlatforms();
  }

  /**
   * Returns a platform object for a given platform index.
   * 
   * @param pPlatformIndex
   *          OpenCL platform index
   * @return platform
   */
  public ClearCLPlatform getPlatform(int pPlatformIndex)
  {
    ClearCLPeerPointer lPlatformIdPointer =
                                          getBackend().getPlatformPeerPointer(pPlatformIndex);
    return new ClearCLPlatform(this, lPlatformIdPointer);
  }

  /**
   * Returns the first device twhich name constain sthe given substring.
   * 
   * @param pNameSubString
   *          name substring
   * @return device
   */
  public ClearCLDevice getDeviceByName(String pNameSubString)
  {
    ClearCLDevice lClearClDevice =
                                 getBestDevice(DeviceName.subString(pNameSubString));
    return lClearClDevice;
  }

  /**
   * Returns the fastest GPU device available for image processing (OpenCL
   * images). This method does its best to avoid crappy Intel integrated cards
   * (except Iris which is not too shabby).
   * 
   * @return fastest GPU device
   */
  public ClearCLDevice getFastestGPUDeviceForImages()
  {
    ClearCLDevice lClearClDevice =
                                 getBestDevice(DeviceTypeSelector.GPU,
                                               BadDeviceSelector.NotSlowIntegratedIntel,
                                               FastestDeviceSelector.FastestForImages);
    return lClearClDevice;
  }

  /**
   * Returns the fastest GPU device available for buffer processing (OpenCL
   * buffers).
   * 
   * @return fastest GPU device for buffers
   */
  public ClearCLDevice getFastestGPUDeviceForBuffers()
  {
    ClearCLDevice lClearClDevice =
                                 getBestDevice(FastestDeviceSelector.FastestForBuffers);
    return lClearClDevice;
  }

  /**
   * Returns the GPU device with most global memory available - hence 'largest'.
   * 
   * @return fastest GPU device
   */
  public ClearCLDevice getLargestGPUDevice()
  {
    ClearCLDevice lClearClDevice =
                                 getBestDevice(DeviceTypeSelector.GPU,
                                               BadDeviceSelector.NotIntegratedIntel,
                                               GlobalMemorySelector.MAX);
    return lClearClDevice;
  }

  /**
   * Returns the best GPU device, right now its the fastest one...
   * 
   * @return best GPU device
   */
  public ClearCLDevice getBestGPUDevice()
  {
    return getFastestGPUDeviceForImages();
  }

  /**
   * Returns the best (right now fastest) CPU device.
   * 
   * @return fastest CPU device
   */
  public ClearCLDevice getBestCPUDevice()
  {
    ClearCLDevice lClearClDevice =
                                 getBestDevice(DeviceTypeSelector.CPU,
                                               FastestDeviceSelector.FastestForImages);
    return lClearClDevice;
  }

  /**
   * Picks the one (first) of the best devices obtained by using the given
   * selectors. Important: The selectors are applied in the order provided. For
   * example, you want first to select GPU devices and then pick the one with
   * the most global memory, not the other way around. The selection algorithm
   * works as follows: for each selector, the list of devices is filtered
   * accordingly, as soon as there is only one device left in the current list
   * of devices, the filtering stops - even if there are more selectors to
   * apply. Some selectors pick a single device from the list (max global mem
   * size), others pick devices of a certain type (e.g. GPU or CPU).
   * 
   * Example:
   * 
   * <pre>
   * getBestDevice(DeviceTypeSelector.GPU,
   *               BadDeviceSelector.NotIntegratedIntel,
   *               GlobalMemorySelector.MAX)
   * </pre>
   * 
   * @param pDeviceSelectors
   *          list of selectors
   * @return best device
   */
  public ClearCLDevice getBestDevice(DeviceSelector... pDeviceSelectors)
  {
    ArrayList<ClearCLDevice> lBestDevices =
                                          getBestDevices(pDeviceSelectors);
    if (lBestDevices.size() > 0)
    {
      ClearCLDevice lFirstBestDevice = lBestDevices.get(0);
      mAccessedDeviceList.add(lFirstBestDevice);
      return lFirstBestDevice;
    }
    else
      return null;
  }

  /**
   * Returns a list of 'best devices' obtained by taking the list of all devices
   * for all platforms and applying the selectors as filters. Important: The
   * selectors are applied in the order provided. For example, you want first to
   * select GPU devices and then pick the one with the most global memory, not
   * the other way around. The selection algorithm works as follows: for each
   * selector, the list of devices is filtered accordingly, as soon as there is
   * only one device left in the current list of devices, the filtering stops -
   * even if there are more selectors to apply. Some selectors pick a single
   * device from the list (max global mem size), others pick devices of a
   * certain type (e.g. GPU or CPU).
   * 
   * @param pDeviceSelectors
   *          vararg list of device selectors
   * @return list of selected devices
   */
  public ArrayList<ClearCLDevice> getBestDevices(DeviceSelector... pDeviceSelectors)
  {

    ArrayList<ClearCLDevice> lSelectedDeviceList = getAllDevices();

    for (DeviceSelector lDeviceSelector : pDeviceSelectors)
    {
      if (lSelectedDeviceList.size() == 1)
        break;

      // System.out.println(lDeviceSelector.getClass().getName());
      // System.out.println(lSelectedDeviceList);
      lDeviceSelector.init(lSelectedDeviceList);

      ArrayList<ClearCLDevice> lTempSelectedList = new ArrayList<>();
      for (ClearCLDevice lDevice : lSelectedDeviceList)
      {
        boolean lSelected = lDeviceSelector.selected(lDevice);
        if (lSelected)
          lTempSelectedList.add(lDevice);
      }

      lSelectedDeviceList.clear();
      lSelectedDeviceList.addAll(lTempSelectedList);
    }

    return lSelectedDeviceList;
  }

  /**
   * Returns all devices for all platforms.
   * 
   * @return all devices list
   */
  public ArrayList<ClearCLDevice> getAllDevices()
  {
    ArrayList<ClearCLDevice> lSelectedDeviceList = new ArrayList<>();

    int lNumberOfPlatforms = getNumberOfPlatforms();
    for (int i = 0; i < lNumberOfPlatforms; i++)
    {
      ClearCLPlatform lPlatform = getPlatform(i);
      int lNumberOfDevices = lPlatform.getNumberOfDevices();
      for (int d = 0; d < lNumberOfDevices; d++)
      {
        ClearCLDevice lClearClDevice = lPlatform.getDevice(d);
        lSelectedDeviceList.add(lClearClDevice);
      }
    }

    mAccessedDeviceList.addAll(lSelectedDeviceList);

    return lSelectedDeviceList;
  }

  @Override
  public void close()
  {
    if (!mAccessedDeviceList.isEmpty())
    {
      for (ClearCLDevice lClearCLDevice : mAccessedDeviceList)
        lClearCLDevice.close();
      mAccessedDeviceList.clear();
    }
  }

}
