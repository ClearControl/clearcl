package clearcl;

import clearcl.abs.ClearCLBase;
import clearcl.enums.DeviceInfo;
import clearcl.enums.DeviceType;
import clearcl.exceptions.ClearCLTooManyContextsException;
import clearcl.recycling.ClearCLRecyclablePeerPointer;
import clearcl.recycling.ClearCLRecyclableRequest;
import coremem.recycling.BasicRecycler;
import coremem.recycling.RecyclableFactoryInterface;
import coremem.recycling.RecyclerInterface;

/**
 * ClearCLDevice is the ClearCL abstraction for OpenCl devices.
 *
 * @author royer
 */
public class ClearCLDevice extends ClearCLBase
{
  private static final int cMaxContexts = 64;

  private final ClearCLPlatform mClearCLPlatform;
  private final ClearCLPeerPointer mDevicePointer;
  private final RecyclerInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest> mContextRecycler;

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

    // Factory that creates new recyclable peer pointers for contexts:
    final RecyclableFactoryInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest> lRecyclableContextPeerPointerFactory =
                                                                                                                                  new RecyclableFactoryInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest>()
                                                                                                                                  {
                                                                                                                                    @Override
                                                                                                                                    public ClearCLRecyclablePeerPointer create(ClearCLRecyclableRequest pParameters)
                                                                                                                                    {

                                                                                                                                      ClearCLPeerPointer lContextPointer =
                                                                                                                                                                         getBackend().getContextPeerPointer(mClearCLPlatform.getPeerPointer(),
                                                                                                                                                                                                            mDevicePointer);
                                                                                                                                      return new ClearCLRecyclablePeerPointer(lContextPointer,
                                                                                                                                                                              ClearCLContext.class);

                                                                                                                                    }
                                                                                                                                  };

    // Recycler that keeps tracks of recyclable peer pointers for contexts:
    mContextRecycler =
                     new BasicRecycler<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest>(lRecyclableContextPeerPointerFactory,
                                                                                               cMaxContexts,
                                                                                               cMaxContexts,
                                                                                               false);

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
    String lStringVersion =
                          getBackend().getDeviceVersion(mDevicePointer)
                                      .replace("OpenCL C", "")
                                      .trim();
    Double lDoubleVersion = Double.parseDouble(lStringVersion);
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
   * Creates device context. Contexts are mannaged internally by a recycler
   * (CoreMem recycler), and are automatically released back to the recycler
   * just before garbage collection. So in theory, there is no need to manually
   * release a context as long as its reference is forgotten.
   * <p>
   * Note: There is a maximal number of contexts that can be created. This is a
   * large number that should accomodate any reasonable application. An
   * exception is thrown by this method if we run out of contexts, which can
   * only happen if they are all used (reference is still known and thus not
   * garbage collected).
   *
   * @return context
   */
  public ClearCLContext createContext()
  {
    // We try to get a recycler, if there is none available (leak) we get null:
    final ClearCLRecyclablePeerPointer lContextRecyclablePointer =
                                                                 mContextRecycler.getOrFail(new ClearCLRecyclableRequest(this,
                                                                                                                         ClearCLContext.class));

    if (lContextRecyclablePointer == null)
      throw new ClearCLTooManyContextsException();

    ClearCLContext lClearCLContext =
                                   new ClearCLContext(this,
                                                      lContextRecyclablePointer);

    return lClearCLContext;
  }

  /**
   * Returns the number of 'live' contexts, i.e. contexts that have been created
   * but not yet released.
   *
   * @return number of live contexts
   */
  public int getNumberOfLiveContexts()
  {
    return mContextRecycler.getNumberOfLiveObjects();
  }

  /**
   * Returns the number of 'available' contexts, i.e. contexts that have been
   * created and the released, and that are now available for recycling
   *
   * @return number of available contexts
   */
  public int getNumberOfAvailableContexts()
  {
    return mContextRecycler.getNumberOfAvailableObjects();
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
    try
    {
      if (getPeerPointer() != null)
      {
        mContextRecycler.clearReleased();

        getBackend().releaseDevice(getPeerPointer());
        setPeerPointer(null);
      }
    }
    catch (Throwable e)
    {
      e.printStackTrace();
    }
  }

}
