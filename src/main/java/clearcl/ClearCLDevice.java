package clearcl;

import clearcl.abs.ClearCLBase;
import clearcl.enums.DeviceInfo;
import clearcl.enums.DeviceType;

/**
 * ClearCLDevice is the ClearCL abstraction for OpenCl devices.
 *
 * @author royer
 */
public class ClearCLDevice extends ClearCLBase
{

  private ClearCLPlatform mClearCLPlatform;
  private ClearCLPeerPointer mDevicePointer;

  /**
   * Construction of this object is done from within a ClearCLPlatform.
   * 
   * @param pClearCLPlatform
   * @param pDevicePointer
   */
  ClearCLDevice(ClearCLPlatform pClearCLPlatform,
                ClearCLPeerPointer pDevicePointer)
  {
    super(pClearCLPlatform.getBackend(), pDevicePointer);
    mClearCLPlatform = pClearCLPlatform;
    mDevicePointer = pDevicePointer;
  }

  /**
   * Returns device name.
   * 
   * @return device name.
   */
  public String getName()
  {
    return getBackend().getDeviceName(mDevicePointer).trim();
  }

  /**
   * Returns device type.
   * 
   * @return device type
   */
  public DeviceType getType()
  {
    return getBackend().getDeviceType(mDevicePointer);
  }

  /**
   * Returns OpenCL version
   * 
   * @return OpenCL version
   */
  public double getVersion()
  {
    final String[] lStringVersion =
                                  getBackend().getDeviceVersion(mDevicePointer)
                                              .replace("OpenCL C", "")
                                              .trim()
                                              .split("\\s+");

    if (lStringVersion.length == 0)
    {
      return 0.;
    }

    Double lDoubleVersion = Double.parseDouble(lStringVersion[0]);
    return lDoubleVersion;
  }

  /**
   * Returns device OpenL extensions string.
   * 
   * @return extensions string
   */
  public String getExtensions()
  {
    return getBackend().getDeviceExtensions(mDevicePointer);
  }

  /**
   * Returns this device global memory size in bytes.
   * 
   * @return global memory size in bytes
   */
  public long getGlobalMemorySizeInBytes()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.MaxGlobalMemory);
  }

  /**
   * Returns this device max memory allocation size.
   * 
   * @return max allocation size
   */
  public long getMaxMemoryAllocationSizeInBytes()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.MaxMemoryAllocationSize);
  }

  /**
   * Returns this device local memory size.
   * 
   * @return local memory size
   */
  public long getLocalMemorySizeInBytes()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.LocalMemSize);
  }

  /**
   * Returns this device clock frequency.
   * 
   * @return clock frequency in MHz
   */
  public long getClockFrequency()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.MaxClockFreq);
  }

  /**
   * Returns this device number of compute units.
   * 
   * @return number of compute units
   */
  public long getNumberOfComputeUnits()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.ComputeUnits);
  }

  /**
   * Returns the max work group size. This means that the product of the local
   * sizes cannot be bigger that this value.
   * 
   * @return max work group size
   */
  public long getMaxWorkGroupSize()
  {
    return getBackend().getDeviceInfo(mDevicePointer,
                                      DeviceInfo.MaxWorkGroupSize);
  }

  /**
   * Returns device info string.
   * 
   * @return device info string
   */
  public String getInfoString()
  {
    StringBuilder lStringBuilder = new StringBuilder();

    lStringBuilder.append(String.format("Device name: %s, type: %s, OpenCL version: %g \n max global memory: %d \n max local memory: %d \n clock freq: %dMhz \n nb compute units: %d \n extensions: %s  \n",
                                        getName(),
                                        getType(),
                                        getVersion(),
                                        getGlobalMemorySizeInBytes(),
                                        getLocalMemorySizeInBytes(),
                                        getClockFrequency(),
                                        getNumberOfComputeUnits(),
                                        getExtensions()));

    return lStringBuilder.toString();
  }

  /**
   * Creates device context.
   * 
   * @return context
   */
  public ClearCLContext createContext()
  {
    ClearCLPeerPointer lContextPointer =
                                       getBackend().getContextPeerPointer(mClearCLPlatform.getPeerPointer(),
                                                                          mDevicePointer);
    ClearCLContext lClearCLContext =
                                   new ClearCLContext(this,
                                                      lContextPointer);

    return lClearCLContext;
  }

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString()
  {
    return String.format("ClearCLDevice [mClearCLPlatform=%s, name=%s]",
                         mClearCLPlatform.toString(),
                         getName());
  }

  /* (non-Javadoc)
   * @see clearcl.ClearCLBase#close()
   */
  @Override
  public void close()
  {
    if (getPeerPointer() != null)
    {
      // not needed as the device was not created through clCreateSubDevices
      // for details, see
      // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clReleaseDevice.html
      // getBackend().releaseDevice(getPeerPointer());
      setPeerPointer(null);
    }
  }

}
