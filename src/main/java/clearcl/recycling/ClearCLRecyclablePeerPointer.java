package clearcl.recycling;

import java.util.concurrent.atomic.AtomicBoolean;

import clearcl.ClearCLPeerPointer;
import coremem.exceptions.FreedException;
import coremem.recycling.RecyclableInterface;
import coremem.recycling.RecyclerInterface;

/**
 * Recyclable peer pointer.
 * <p>
 * Peer pointers of this kind can be recycled. This is usefull in the case of
 * tricky ressources (for example contexts) that don't behave well when created
 * or destroyed. using this type of recyclable peer pointer reduces the risks
 * and system pressure associated with creating such ressources.
 * <p>
 * Note: Best practice is to avoid mixing different types of pointer in the same
 * recycler. Actually, just don't do that, at all (why would you anyway, think
 * about it).
 */
public class ClearCLRecyclablePeerPointer extends ClearCLPeerPointer
                                          implements
                                          RecyclableInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest>
{

  private RecyclerInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest> mRecycler;

  private final AtomicBoolean mReleased = new AtomicBoolean(true);
  private final Class<?> mClass;

  /**
   * Constructs a recyclable peer pointer given an existing peer pointer. The
   * actual pointer information is copied. A class must be provided in order to
   * know whih
   *
   * @param pPeerPointer
   *          peer pointer to use
   * @param pClass
   *          type
   */
  public ClearCLRecyclablePeerPointer(ClearCLPeerPointer pPeerPointer,
                                      Class<?> pClass)
  {
    super(pPeerPointer.getPointer());
    mClass = pClass;
  }

  @Override
  public boolean isCompatible(ClearCLRecyclableRequest pRequest)
  {
    return pRequest.mClass.equals(mClass);
  }

  @Override
  public void recycle(ClearCLRecyclableRequest pRequest)
  {
    // nothing to do as we just use the same pointer contained here.
  }

  @Override
  public void setRecycler(RecyclerInterface<ClearCLRecyclablePeerPointer, ClearCLRecyclableRequest> pRecycler)
  {
    mRecycler = pRecycler;
  }

  @Override
  public void setReleased(boolean pIsReleased)
  {
    mReleased.set(pIsReleased);
  }

  @Override
  public boolean isReleased()
  {
    return mReleased.get();
  }

  @Override
  public void release()
  {
    mRecycler.release(this);
  }

  @Override
  public long getSizeInBytes()
  {
    return 0;
  }

  @Override
  public void free()
  {
  }

  @Override
  public boolean isFree()
  {
    return false;
  }

  @Override
  public void complainIfFreed() throws FreedException
  {

  }

}