package clearcl.exceptions;

/**
 * Exception thrown when too many contexts have been created and not released
 *
 * @author royer
 */
public class ClearCLTooManyContextsException extends ClearCLException
{

  private static final long serialVersionUID = 1L;

  /**
   * Instanciates this exception
   */
  public ClearCLTooManyContextsException()
  {
    super("Too many contexts have been created and not released");
  }

  /**
   * Instanciates this exception with a given message
   *
   * @param pMessage
   *          message
   */
  public ClearCLTooManyContextsException(String pMessage)
  {
    super("Too many contexts have been created and not released: "
          + pMessage);
  }

}
