
package clearcl.ops.kernels;

/**
 *
 * @author nico
 */
public class CLKernelException extends Exception
{
  private String kernelSourceCode_;

  public CLKernelException()
  {
    super();
  }

  public CLKernelException(String message)
  {
    super(message);
  }

  public CLKernelException(String message, String srcCode)
  {
    super(message);
    kernelSourceCode_ = srcCode;
  }

  public String getKernelSourceCode()
  {
    return kernelSourceCode_;
  }

}