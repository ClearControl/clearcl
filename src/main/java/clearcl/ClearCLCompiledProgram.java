package clearcl;

import java.util.concurrent.ConcurrentHashMap;

import clearcl.abs.ClearCLBase;
import clearcl.backend.ClearCLBackendInterface;
import clearcl.enums.BuildStatus;
import coremem.rgc.Cleanable;
import coremem.rgc.Cleaner;
import coremem.rgc.RessourceCleaner;

/**
 * ClearCLCompiledProgram is an internal abstraction for compiled programs.
 * <p>
 * Note: This object did not exist in the early versions of ClearCL. This was
 * introduced to fix the RGC bug that caused ressource leakage. This object
 * better matches the lifecycle of the compiled OpenCL program and thus is
 * better suited for properly supporting ressource cleanup at GC time.
 *
 *
 * @author royer
 */
public class ClearCLCompiledProgram extends ClearCLBase
                                    implements Cleanable
{

  private final ClearCLDevice mDevice;
  private final ClearCLContext mContext;
  private final ClearCLProgram mClearCLProgram;
  private final String mSourceCode;

  private ConcurrentHashMap<String, ClearCLKernel> mKernelCache =
                                                                new ConcurrentHashMap<String, ClearCLKernel>();

  /**
   * This constructor is called internally from a program.
   *
   * @param pDevice
   * @param pClearCLContext
   * @param pProgramPointer
   */
  ClearCLCompiledProgram(ClearCLDevice pDevice,
                         ClearCLContext pClearCLContext,
                         ClearCLProgram pClearCLProgram,
                         String pSourceCode,
                         ClearCLPeerPointer pProgramPointer)
  {
    super(pClearCLContext.getBackend(), pProgramPointer);
    mDevice = pDevice;
    mContext = pClearCLContext;
    mClearCLProgram = pClearCLProgram;
    mSourceCode = pSourceCode;

    // This will register this compiled program for GC cleanup
    if (ClearCL.sRGC)
      RessourceCleaner.register(this);
  }

  /**
   * Returns the device for this program.
   *
   * @return device
   */
  public ClearCLDevice getDevice()
  {
    return mDevice;
  }

  /**
   * Returns the context for this program.
   *
   * @return context
   */
  public ClearCLContext getContext()
  {
    return mContext;
  }

  /**
   * Returns last build status for this program.
   *
   * @return last build status
   */
  public BuildStatus getBuildStatus()
  {
    BuildStatus lBuildStatus =
                             getBackend().getBuildStatus(getDevice().getPeerPointer(),
                                                         getPeerPointer());
    return lBuildStatus;
  }

  /**
   * Returns last build logs for this program.
   *
   * @return build logs
   */
  public String getBuildLog()
  {
    String lBuildLog = getBackend()
                                   .getBuildLog(getDevice().getPeerPointer(),
                                                getPeerPointer())
                                   .trim();
    return lBuildLog;
  }

  /**
   * Returns the kernel of a given name. If the kernel is not yet created, it is
   * then created on-demand. Note: this will not recreate a kernel that already
   * exists.
   *
   * @param pKernelName
   *          kernel name
   * @return kernel
   */
  public ClearCLKernel getKernel(String pKernelName)
  {
    ClearCLKernel lClearCLKernel = mKernelCache.get(pKernelName);
    if (lClearCLKernel == null)
    {
      lClearCLKernel = createKernel(pKernelName);
    }
    return lClearCLKernel;
  }

  /**
   * Creates kernel of given name from this program.
   *
   * @param pKernelName
   *          kernel name (function name)
   * @return kernel
   */
  public ClearCLKernel createKernel(String pKernelName)
  {

    ClearCLPeerPointer lKernelPointer =
                                      getBackend().getKernelPeerPointer(this.getPeerPointer(),
                                                                        pKernelName);

    ClearCLKernel lClearCLKernel = new ClearCLKernel(getContext(),
                                                     this,
                                                     lKernelPointer,
                                                     pKernelName,
                                                     mSourceCode);

    mKernelCache.put(pKernelName, lClearCLKernel);

    return lClearCLKernel;

  }

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString()
  {
    return String.format("ClearCLCompiledProgram[%s]",
                         mClearCLProgram.toString());
  }

  /* (non-Javadoc)
   * @see clearcl.ClearCLBase#close()
   */
  @Override
  public void close()
  {

    if (getPeerPointer() != null)
    {
      if (mCompiledProgramCleaner != null)
        mCompiledProgramCleaner.mClearCLPeerPointer = null;
      getBackend().releaseProgram(getPeerPointer());
      setPeerPointer(null);
    }

  }

  // NOTE: this _must_ be a static class, otherwise instances of this class will
  // implicitely hold a reference of this image...
  private static class CompiledProgramCleaner implements Cleaner
  {
    public ClearCLBackendInterface mBackend;
    public volatile ClearCLPeerPointer mClearCLPeerPointer;

    public CompiledProgramCleaner(ClearCLBackendInterface pBackend,
                                  ClearCLPeerPointer pClearCLPeerPointer)
    {
      mBackend = pBackend;
      mClearCLPeerPointer = pClearCLPeerPointer;
    }

    @Override
    public void run()
    {

      try
      {
        if (mClearCLPeerPointer != null)
        {
          if (ClearCL.sDebugRGC)
            System.out.println("Releasing compiled program: "
                               + mClearCLPeerPointer);
          mBackend.releaseProgram(mClearCLPeerPointer);
          mClearCLPeerPointer = null;
        }
      }
      catch (Throwable e)
      {
        if (ClearCL.sDebugRGC)
          e.printStackTrace();
      }
    }

  }

  CompiledProgramCleaner mCompiledProgramCleaner;

  @Override
  public Cleaner getCleaner()
  {
    mCompiledProgramCleaner =
                            new CompiledProgramCleaner(getBackend(),
                                                       getPeerPointer());

    return mCompiledProgramCleaner;
  }

}
