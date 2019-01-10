package clearcl.backend.javacl.test;

import static org.junit.Assert.assertTrue;

import clearcl.backend.ClearCLBackendInterface;
import clearcl.backend.javacl.ClearCLBackendJavaCL;

import org.junit.Test;

/**
 * 
 *
 * @author royer
 */
public class ClearCLBackendJavaCLTests
{

  /**
   * 
   */
  @Test
  public void test()
  {
    ClearCLBackendInterface lClearCLBackend =
                                            new ClearCLBackendJavaCL();

    assertTrue(lClearCLBackend.getNumberOfPlatforms() > 0);

  }

}
