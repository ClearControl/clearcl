package clearcl;

/**
 * ClearCLKernel is the ClearCL abstraction for objects used by backends to wrap
 * OpenCL object pointers.
 * 
 * @author royer
 */
public class ClearCLPeerPointer
{
  protected final Object mPointer;

  /**
   * Creates a peer pointer from a backend internal pointer-wrapping object.
   * 
   * @param pPointer
   *          pointer object
   */
  public ClearCLPeerPointer(Object pPointer)
  {
    super();
    mPointer = pPointer;
  }

  /**
   * Returns backend-internal pointer-wrapping object.
   * 
   * @return backend-internal pointer-wrapping object.
   */
  public Object getPointer()
  {
    return mPointer;
  }

}
