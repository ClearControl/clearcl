package clearcl;

import static org.junit.Assert.*;

import clearcl.backend.jocl.ClearCLBackendJOCL;

import org.junit.Test;

public class ClearCLTest
{
  @Test
  public void getAllDevicesTest()
  {
    ClearCL clearCL = new ClearCL(new ClearCLBackendJOCL());
    for (ClearCLDevice device : clearCL.getAllDevices())
    {
      System.out.println("Device name :" + device.getName());
    }
  }
}