package clearcl;

import java.lang.ref.WeakReference;

import clearcl.abs.ClearCLBase;
import clearcl.backend.ClearCLBackendInterface;
import coremem.rgc.Cleanable;
import coremem.rgc.Cleaner;
import coremem.rgc.RessourceCleaner;

/**
 * ClearCLQueue is the ClearCL abstraction for OpenCl queues.
 *
 * @author royer
 */
public class ClearCLQueue extends ClearCLBase implements Cleanable
{

  private WeakReference<ClearCLContext> mClearCLContextReference;

  /**
   * This constructor is called internally from an OpenCl context.
   *
   * @param pClearCLContext
   *          context
   * @param pQueuePointer
   *          queue peer pointer
   */
  public ClearCLQueue(ClearCLContext pClearCLContext,
                      ClearCLPeerPointer pQueuePointer)
  {
    super(pClearCLContext.getBackend(), pQueuePointer);
    mClearCLContextReference = new WeakReference<>(pClearCLContext);

    // This will register this queue for GC cleanup
    if (ClearCL.sRGC)
      RessourceCleaner.register(this);
  }

  /**
   * Returns this queues context.
   *
   * @return context
   */
  public ClearCLContext getContext()
  {
    return mClearCLContextReference.get();
  }

  /**
   * Waits for queue to finish enqueued tasks (such as: kernel execution, buffer
   * and image copies, writes and reads).
   */
  public void waitToFinish()
  {
    getBackend().waitQueueToFinish(getPeerPointer());
  }

  @Override
  public void close()
  {
    if (getPeerPointer() != null)
    {
      if (mQueueCleaner != null)
        mQueueCleaner.mClearCLPeerPointer = null;
      getBackend().releaseQueue(getPeerPointer());
      setPeerPointer(null);
    }
  }

  // NOTE: this _must_ be a static class, otherwise instances of this class will
  // implicitely hold a reference of this image...
  private static class QueueCleaner implements Cleaner
  {
    public ClearCLBackendInterface mBackend;
    public volatile ClearCLPeerPointer mClearCLPeerPointer;

    public QueueCleaner(ClearCLBackendInterface pBackend,
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
            System.out.println("Releasing queue:   "
                               + mClearCLPeerPointer.toString());

          mBackend.releaseQueue(mClearCLPeerPointer);
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

  QueueCleaner mQueueCleaner;

  @Override
  public Cleaner getCleaner()
  {
    mQueueCleaner = new QueueCleaner(getBackend(), getPeerPointer());
    return mQueueCleaner;
  }
}
