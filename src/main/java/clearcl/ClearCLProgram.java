package clearcl;

import java.io.*;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import clearcl.enums.BuildStatus;
import clearcl.enums.ImageChannelDataType;
import clearcl.exceptions.ClearCLProgramNotBuiltException;
import clearcl.ocllib.OCLlib;
import clearcl.util.StringUtils;

/**
 * ClearCLProgram is the ClearCL abstraction for OpenCl programs.
 *
 * @author royer
 */
public class ClearCLProgram implements AutoCloseable
{
  private final ClearCLDevice mDevice;
  private final ClearCLContext mContext;
  private final ArrayList<String> mSourceCode;
  private final ConcurrentHashMap<String, String> mDefinesMap =
                                                              new ConcurrentHashMap<>();
  private final ArrayList<String> mBuildOptionsList =
                                                    new ArrayList<>();
  private final ArrayList<String> mIncludesSearchPackages =
                                                          new ArrayList<>();

  private volatile boolean mModified = true;
  private volatile String mLastBuiltSourceCode;

  private ClearCLCompiledProgram mCompiledProgram;

  /**
   * This constructor is called internally from an OpenCl context.
   *
   * @param pDevice
   * @param pClearCLContext
   */
  ClearCLProgram(ClearCLDevice pDevice,
                 ClearCLContext pClearCLContext)
  {
    mDevice = pDevice;
    mContext = pClearCLContext;
    mSourceCode = new ArrayList<String>();
    addIncludesSearchPackage("clearcl.ocllib");
  }

  /**
   * Adds a package to the list of Java packages to search for reference classes
   * when loading includes in cl files.
   *
   * @param pPackagePath
   *          full package path and package name.
   */
  public void addIncludesSearchPackage(String pPackagePath)
  {
    mIncludesSearchPackages.add(pPackagePath);
    mModified = true;
  }

  /**
   * Adds a reference class used to locate resources (*.cl files) for includes
   * in cl files. This method is preferred because it is more resiliant to
   * refactoring.
   *
   * @param pReferenceClass
   *          reference class
   */
  public void addIncludesReferenceClass(Class<?> pReferenceClass)
  {
    String lPackageName = pReferenceClass.getPackage().getName();
    mIncludesSearchPackages.add(lPackageName);
    mModified = true;
  }

  /**
   * Adds source code to this program. You must rebuild after this call for
   * changes to take effect.
   *
   * @param pSourceCode
   *          source code string
   */
  public void addSource(String pSourceCode)
  {
    mSourceCode.add(pSourceCode);
    mModified = true;
  }

  /**
   * Adds source code to this program from a resource file located relative to a
   * given class. You must rebuild after this call for changes to take effect.
   *
   * @param pClassForRessource
   *          Reference class to locate resources
   * @param pIncludeRessourceName
   *          Resource file names (relative to reference class)
   * @throws IOException
   *           if problem while loading resource
   */
  public void addSource(Class<?> pClassForRessource,
                        String pIncludeRessourceName) throws IOException
  {
    StringBuilder lStringBuilder = new StringBuilder();

    lStringBuilder.append("\n\n");
    lStringBuilder.append(" //###########################################################################\n");
    lStringBuilder.append("// Source: '" + pIncludeRessourceName
                          + "' relative to "
                          + pClassForRessource.getSimpleName()
                          + "\n");

    InputStream lResourceAsStream =
                                  pClassForRessource.getResourceAsStream(pIncludeRessourceName);
    if (lResourceAsStream == null)
    {
      // split packages of a classname
      String[] lPath = pClassForRessource.getCanonicalName()
                                         .split("\\.");
      // keep all but the last one
      String[] lCutPath = new String[lPath.length - 1];
      System.arraycopy(lPath, 0, lCutPath, 0, lCutPath.length);

      // assemble new filename
      String lAbsoluteKernelFilename =
                                     ("/" + String.join("/", lCutPath)
                                      + "/"
                                      + pIncludeRessourceName).replace("/./",
                                                                       "/");
      lResourceAsStream =
                        pClassForRessource.getResourceAsStream(lAbsoluteKernelFilename);

      // In case we are searching in the wrong root:
      if (lResourceAsStream == null)
      {
        lAbsoluteKernelFilename.replace("/test/", "/main/");
        lResourceAsStream =
                          pClassForRessource.getResourceAsStream(lAbsoluteKernelFilename);
      }
    }
    if (lResourceAsStream == null)
    {
      File lKernelFile = new File(pIncludeRessourceName);
      if (lKernelFile.exists())
      {
        try
        {
          lResourceAsStream = new FileInputStream(lKernelFile);
        }
        catch (FileNotFoundException e)
        {
          e.printStackTrace();
        }
      }
    }
    if (lResourceAsStream == null)
    {
      String lMessage = String.format("Cannot find source: [%s] %s",
                                      pClassForRessource.getSimpleName()
                                                        .toString(),
                                      pIncludeRessourceName);
      throw new IOException(lMessage);
    }

    String lSourceCode = StringUtils.streamToString(lResourceAsStream,
                                                    "UTF-8");
    lStringBuilder.append(lSourceCode);
    lStringBuilder.append("\n\n");

    addSource(lStringBuilder.toString());
  }

  /**
   * Clears all sources in this program. You must rebuild after this call for
   * changes to take effect.
   */
  public void clearSources()
  {
    mSourceCode.clear();
    mModified = true;
  }

  /**
   * Adds a define (e.g. #define 'key' 'value') to the program source code. You
   * must rebuild after this call for changes to take effect.
   *
   * @param pKey
   *          key
   * @param pValue
   *          value
   */
  public void addDefine(String pKey, String pValue)
  {
    mDefinesMap.put(pKey, pValue);
    mModified = true;
  }

  /**
   * Adds a numerical define (e.g. #define 'key' 'value' with 'value' a number)
   * to the program source code. You must rebuild after this call for changes to
   * take effect.
   *
   * @param pKey
   *          key
   * @param pValue
   *          value
   */
  public void addDefine(String pKey, Number pValue)
  {
    if (pValue instanceof Byte || pValue instanceof Short
        || pValue instanceof Integer
        || pValue instanceof Long)
      mDefinesMap.put(pKey, "" + pValue.longValue());
    else if (pValue instanceof Float)
      mDefinesMap.put(pKey,
                      String.format(Locale.US,
                                    "%ef",
                                    pValue.floatValue()));
    else if (pValue instanceof Double)
      mDefinesMap.put(pKey,
                      String.format(Locale.US,
                                    "%ef",
                                    pValue.doubleValue()));

    mModified = true;
  }

  /**
   * Adds a define (e.g. #define 'symbol') to the program source code. You must
   * rebuild after this call for changes to take effect.
   *
   * @param pSymbol
   *          define symbol
   */
  public void addDefine(String pSymbol)
  {
    mDefinesMap.put(pSymbol, "");
    mModified = true;
  }

  /**
   * Adds one of three kinds of defines: FLOAT, INT and UINT depending on the
   * given channel datatype. This is usefull for kernels that need to switch
   * between different variants of write_imageX and read_imageX.
   *
   * @param pChannelDataType
   *          channel data type
   */
  public void addDefineForDataType(ImageChannelDataType pChannelDataType)
  {
    switch (pChannelDataType)
    {
    case Float:
      addDefine("FLOAT");
      break;
    case HalfFloat:
      addDefine("FLOAT");
      break;
    case SignedInt16:
      addDefine("INT");
      break;
    case SignedInt32:
      addDefine("INT");
      break;
    case SignedInt8:
      addDefine("INT");
      break;
    case SignedNormalizedInt16:
      addDefine("FLOAT");
      break;
    case SignedNormalizedInt8:
      addDefine("FLOAT");
      break;
    case UnsignedInt16:
      addDefine("UINT");
      break;
    case UnsignedInt32:
      addDefine("UINT");
      break;
    case UnsignedInt8:
      addDefine("UINT");
      break;
    case UnsignedNormalizedInt16:
      addDefine("FLOAT");
      break;
    case UnsignedNormalizedInt8:
      addDefine("FLOAT");
      break;
    default:
      break;

    }

  }

  /**
   * Clears all defines. You must rebuild after this call for changes to take
   * effect.
   */
  public void clearDefines()
  {
    mDefinesMap.clear();
    mModified = true;
  }

  /**
   * Adds an building option for this program. You must rebuild after this call
   * for changes to take effect.
   *
   * @param pOption
   *          option string
   */
  public void addBuildOption(String pOption)
  {
    mBuildOptionsList.add(pOption);
    mModified = true;
  }

  /**
   * Clears all options for this program. You must rebuild after this call for
   * changes to take effect.
   */
  public void clearBuildOptions()
  {
    mBuildOptionsList.clear();
    mModified = true;
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
   * Builds program and logs any errors on the stdout
   *
   * @return build status
   * @throws IOException
   *           thrown if source code includes cannot be resolved
   */
  public BuildStatus buildAndLog() throws IOException
  {
    BuildStatus lBuildStatus = build();
    if (lBuildStatus != BuildStatus.Success)
    {
      System.out.println(getSourceCode());
      System.err.println(getBuildLog());
    }
    return lBuildStatus;
  }

  /**
   * Builds this program. The source code can be changed after a first build and
   * this method will build a new program from scratch.
   *
   * @return build status
   * @throws IOException
   *           thrown if source code includes cannot be resolved
   */
  public BuildStatus build() throws IOException
  {

    // If nothing changed, don't build.
    if (!mModified)
      return getBuildStatus();

    // Get the source:
    try
    {
      mLastBuiltSourceCode = getSourceCode();
    }
    catch (Throwable e)
    {
      e.printStackTrace();
      return BuildStatus.Error;
    }

    ClearCLPeerPointer lProgramPeerPointer =
                                           mContext.getBackend()
                                                   .getProgramPeerPointer(mContext.getPeerPointer(),
                                                                          mLastBuiltSourceCode);

    String lOptions = concatenateOptions();
    mContext.getBackend().buildProgram(lProgramPeerPointer, lOptions);

    mCompiledProgram =
                     new ClearCLCompiledProgram(mDevice,
                                                mContext,
                                                this,
                                                mLastBuiltSourceCode,
                                                lProgramPeerPointer);

    mModified = false;

    return getBuildStatus();
  }

  /**
   * Returns the complete concatenated source code with includes and defines
   * added.
   *
   * @return source code
   * @throws IOException
   *           thrown if the includes cannot be resolved
   */
  public String getSourceCode() throws IOException
  {
    String lConcatenatedSourceCode = concatenateSourceCode();

    String lSourceCodeWithDefines =
                                  insertDefines(lConcatenatedSourceCode);
    String lSourceCodeWithDefinesAndIncludes =
                                             insertIncludes(lSourceCodeWithDefines);
    String lSourceCodeWithPreamble =
                                   insertPreamble(lSourceCodeWithDefinesAndIncludes);
    return lSourceCodeWithPreamble;
  }

  private String insertPreamble(String pSourceCode) throws IOException
  {
    InputStream lResourceAsStream =
                                  OCLlib.class.getResourceAsStream("preamble/preamble.cl");

    if (lResourceAsStream == null)
    {
      String lMessage =
                      String.format("Cannot find preamble file at 'preamble/preamble.cl'");
      throw new IOException(lMessage);
    }
    String lPreambleCode =
                         StringUtils.streamToString(lResourceAsStream,
                                                    "UTF-8");

    StringBuilder lStringBuilder = new StringBuilder();

    lStringBuilder.append("\n\n");
    lStringBuilder.append(" //###########################################################################\n");
    lStringBuilder.append("// Preamble:\n");
    lStringBuilder.append(lPreambleCode);
    lStringBuilder.append("\n\n");
    lStringBuilder.append(pSourceCode);

    return lStringBuilder.toString();
  }

  private String insertDefines(String pSourceCode)
  {
    StringBuilder lStringBuilder = new StringBuilder();

    lStringBuilder.append("\n\n");
    lStringBuilder.append(" //###########################################################################\n");
    lStringBuilder.append("// Defines:\n");
    for (Map.Entry<String, String> lDefinesEntry : mDefinesMap.entrySet())
    {

      if (lDefinesEntry.getValue().length() == 0)
        lStringBuilder.append("#define " + lDefinesEntry.getKey()
                              + "\n");
      else
        lStringBuilder.append("#define " + lDefinesEntry.getKey()
                              + " \t"
                              + lDefinesEntry.getValue()
                              + "\n");
    }
    lStringBuilder.append(pSourceCode);

    String lDefinesAndSourceCode = lStringBuilder.toString();
    return lDefinesAndSourceCode;
  }

  private String insertIncludes(String pSourceCode) throws IOException
  {
    StringBuilder lSourceCodeWithIncludes = new StringBuilder();

    int lBeginIndex = 0;
    int lIncludeIndex = 0;
    while ((lIncludeIndex = pSourceCode.indexOf("#include",
                                                lBeginIndex)) >= 0)
    {
      lSourceCodeWithIncludes.append(pSourceCode.substring(lBeginIndex,
                                                           lIncludeIndex));

      int lEndOfLine = pSourceCode.indexOf('\n', lIncludeIndex);

      String lIncludeLine = pSourceCode.substring(lIncludeIndex,
                                                  lEndOfLine);

      int lClassNameBegin = lIncludeLine.indexOf('[');
      int lClassNameEnd = lIncludeLine.indexOf(']',
                                               lClassNameBegin + 1);
      String lClassName = lClassNameBegin == -1 ? ""
                                                : lIncludeLine.substring(lClassNameBegin
                                                                         + 1,
                                                                         lClassNameEnd)
                                                              .trim();

      int lIncludeNameBegin = lIncludeLine.indexOf('"');
      int lIncludeNameEnd =
                          lIncludeLine.indexOf('"',
                                               lIncludeNameBegin + 1);

      String lIncludeName = lIncludeLine
                                        .substring(lIncludeNameBegin
                                                   + 1,
                                                   lIncludeNameEnd)
                                        .trim();

      Class<?> lReferenceClass = Object.class;
      if (lClassName.isEmpty())
        lIncludeName = "/" + lIncludeName;
      else
        lReferenceClass = findClassByName(lClassName);

      if (lReferenceClass == null)
        System.err.println("reference class unknown: " + lClassName
                           + " in line: '"
                           + lIncludeLine
                           + "'");

      InputStream lResourceAsStream =
                                    lReferenceClass.getResourceAsStream(lIncludeName);

      if (lResourceAsStream != null)
      {

        String lSourceCode =
                           StringUtils.streamToString(lResourceAsStream,
                                                      "UTF-8");

        lSourceCodeWithIncludes.append("\n\n");
        lSourceCodeWithIncludes.append(" //___________________________________________________________________________\n");
        lSourceCodeWithIncludes.append("// Begin Include: '"
                                       + lIncludeName + "'\n");
        lSourceCodeWithIncludes.append(lSourceCode);
        lSourceCodeWithIncludes.append("// End Include: '"
                                       + lIncludeName + "'\n");
        lSourceCodeWithIncludes.append(" //___________________________________________________________________________\n");
        lSourceCodeWithIncludes.append("\n\n");

      }
      else
      {
        lSourceCodeWithIncludes.append("\n");
        lSourceCodeWithIncludes.append("//WARNING!! Could not resolve include given below:\n");
        lSourceCodeWithIncludes.append("//" + lIncludeLine);
      }

      lBeginIndex = lEndOfLine;
    }
    lSourceCodeWithIncludes.append(pSourceCode.substring(lBeginIndex));

    return lSourceCodeWithIncludes.toString();
  }

  private Class<?> findClassByName(String pClassName)
  {
    for (String lPackage : mIncludesSearchPackages)
    {
      try
      {
        return Class.forName(lPackage + "." + pClassName);
      }
      catch (ClassNotFoundException e)
      {
        // not in this package, try another
      }
    }
    // nothing found: return null or throw ClassNotFoundException
    return null;
  }

  private String concatenateOptions()
  {
    StringBuilder lStringBuilder = new StringBuilder();
    for (String lOptions : mBuildOptionsList)
    {
      lStringBuilder.append(" ");
      lStringBuilder.append(lOptions);
    }
    return lStringBuilder.toString();
  }

  private String concatenateSourceCode()
  {
    StringBuilder lStringBuilder = new StringBuilder();
    for (String lSourceCode : mSourceCode)
    {
      lStringBuilder.append("\n\n\n");
      lStringBuilder.append(lSourceCode);
    }
    return lStringBuilder.toString();
  }

  /**
   * Returns last build status for this program.
   *
   * @return last build status
   */
  public BuildStatus getBuildStatus()
  {
    if (mCompiledProgram == null || mModified)
      throw new ClearCLProgramNotBuiltException();

    BuildStatus lBuildStatus =
                             mContext.getBackend()
                                     .getBuildStatus(getDevice().getPeerPointer(),
                                                     mCompiledProgram.getPeerPointer());
    return lBuildStatus;
  }

  /**
   * Returns last build logs for this program.
   *
   * @return build logs
   */
  public String getBuildLog()
  {
    if (mCompiledProgram == null || mModified)
      throw new ClearCLProgramNotBuiltException();

    String lBuildLog = mContext.getBackend()
                               .getBuildLog(getDevice().getPeerPointer(),
                                            mCompiledProgram.getPeerPointer())
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
    if (mCompiledProgram == null || mModified)
      throw new ClearCLProgramNotBuiltException();

    return mCompiledProgram.getKernel(pKernelName);
  }

  /**
   * Creates kernel of given name from this program
   *
   * @param pKernelName
   *          kernel name (function name)
   * @return kernel
   */
  public ClearCLKernel createKernel(String pKernelName)
  {
    if (mCompiledProgram == null || mModified)
      throw new ClearCLProgramNotBuiltException();

    return mCompiledProgram.createKernel(pKernelName);
  }

  /**
   * Returns the number of lines of code for this program.
   *
   * @return number of line of code.
   */
  public int getNumberLinesOfCode()
  {
    int lCounter = 0;
    for (String lSourceCodeFile : mSourceCode)
    {
      String[] lSplit = lSourceCodeFile.split("\\r?\\n");
      lCounter += lSplit.length;
    }
    return lCounter;
  }

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
   */
  @Override
  public String toString()
  {
    return String.format("ClearCLProgram [mOptions=%s, mDefinesMap=%s, lines of code:%d]",
                         mBuildOptionsList.toString(),
                         mDefinesMap.toString(),
                         getNumberLinesOfCode());
  }

  /**
   * Adds all math optimizations option available. Note: this might lead to
   * 'unsafe' floating point behavior, although in practice it is rarely a
   * problem.F You must rebuild after this call for changes to take effect.
   */
  public void addBuildOptionAllMathOpt()
  {
    addBuildOptionFastRelaxedMath();
    addBuildOptionFiniteMathOnly();
    addBuildOptionMadEnable();
    addBuildOptionUnsafeMathOptimizations();
  }

  /**
   * Add the -cl-fast-relaxed-math compile option.<br>
   * Sets the optimization options -cl-finite-math-only and
   * -cl-unsafe-math-optimizations. This allows optimizations for floating-point
   * arithmetic that may violate the IEEE 754 standard and the OpenCL numerical
   * compliance requirements defined in the specification in section 7.4 for
   * single-precision floating-point, section 9.3.9 for double-precision
   * floating-point, and edge case behavior in section 7.5. This option causes
   * the preprocessor macro __FAST_RELAXED_MATH__ to be defined in the OpenCL
   * program. <br>
   * Also see : <a href=
   * "http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html"
   * >Khronos' documentation for clBuildProgram</a>.
   */
  public void addBuildOptionFastRelaxedMath()
  {
    addBuildOption("-cl-fast-relaxed-math");
  }

  /**
   * Add the -cl-no-signed-zero compile option.<br>
   * Allow optimizations for floating-point arithmetic that ignore the
   * signedness of zero. IEEE 754 arithmetic specifies the behavior of distinct
   * +0.0 and -0.0 values, which then prohibits simplification of expressions
   * such as x+0.0 or 0.0*x (even with -clfinite-math only). This option implies
   * that the sign of a zero result isn't significant. <br>
   * Also see : <a href=
   * "http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html"
   * >Khronos' documentation for clBuildProgram</a>.
   */
  public void addBuildOptionNoSignedZero()
  {
    addBuildOption("-cl-no-signed-zero");
  }

  /**
   * Add the -cl-mad-enable compile option.<br>
   * Allow a * b + c to be replaced by a mad. The mad computes a * b + c with
   * reduced accuracy. For example, some OpenCL devices implement mad as
   * truncate the result of a * b before adding it to c.<br>
   * Also see : <a href=
   * "http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html"
   * >Khronos' documentation for clBuildProgram</a>.
   */
  public void addBuildOptionMadEnable()
  {
    addBuildOption("-cl-mad-enable");
  }

  /**
   * Add the -cl-finite-math-only compile option.<br>
   * Allow optimizations for floating-point arithmetic that assume that
   * arguments and results are not NaNs or infinites. This option may violate
   * the OpenCL numerical compliance requirements defined in in section 7.4 for
   * single-precision floating-point, section 9.3.9 for double-precision
   * floating-point, and edge case behavior in section 7.5.<br>
   * Also see : <a href=
   * "http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html"
   * >Khronos' documentation for clBuildProgram</a>.
   */
  public void addBuildOptionFiniteMathOnly()
  {
    addBuildOption("-cl-finite-math-only");
  }

  /**
   * Add the -cl-unsafe-math-optimizations option.<br>
   * Allow optimizations for floating-point arithmetic that (a) assume that
   * arguments and results are valid, (b) may violate IEEE 754 standard and (c)
   * may violate the OpenCL numerical compliance requirements as defined in
   * section 7.4 for single-precision floating-point, section 9.3.9 for
   * double-precision floating-point, and edge case behavior in section 7.5.
   * This option includes the -cl-no-signed-zeros and -cl-mad-enable
   * options.<br>
   * Also see : <a href=
   * "http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clBuildProgram.html"
   * >Khronos' documentation for clBuildProgram</a>.
   */
  public void addBuildOptionUnsafeMathOptimizations()
  {
    addBuildOption("-cl-unsafe-math-optimizations");
  }

  /**
   * Add the <a href=
   * "http://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Extensions/cl_nv_compiler_options.txt"
   * >-cl-nv-verbose</a> compilation option (<b><i>NVIDIA GPUs only</i></b>)<br>
   * Enable verbose mode. Output will be reported in JavaCL's log at the INFO
   * level
   */
  public void addBuildOptionNVVerbose()
  {
    addBuildOption("-cl-nv-verbose");
  }

  /**
   * Add the <a href=
   * "http://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Extensions/cl_nv_compiler_options.txt"
   * >-cl-nv-maxrregcount=N</a> compilation option (<b><i>NVIDIA GPUs
   * only</i></b>)<br>
   * Specify the maximum number of registers that GPU functions can use. Until a
   * function-specific limit, a higher value will generally increase the
   * performance of individual GPU threads that execute this function. However,
   * because thread registers are allocated from a global register pool on each
   * GPU, a higher value of this option will also reduce the maximum thread
   * block size, thereby reducing the amount of thread parallelism. Hence, a
   * good maxrregcount value is the result of a trade-off. If this option is not
   * specified, then no maximum is assumed. Otherwise the specified value will
   * be rounded to the next multiple of 4 registers until the GPU specific
   * maximum of 128 registers.
   *
   * @param N
   *          positive integer
   */
  public void addBuildOptionNVMaximumRegistryCount(int N)
  {
    addBuildOption("-cl-nv-maxrregcount=" + N);
  }

  /**
   * Add the <a href=
   * "http://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/OpenCL_Extensions/cl_nv_compiler_options.txt"
   * >-cl-nv-opt-level</a> compilation option (<b><i>NVIDIA GPUs
   * only</i></b>)<br>
   * Specify optimization level (default value: 3)
   *
   * @param N
   *          positive integer, or 0 (no optimization).
   */
  public void addBuildOptionNVOptimizationLevel(int N)
  {
    addBuildOption("-cl-nv-opt-level=" + N);
  }

  @Override
  public void close() throws IOException
  {
    // nothing to do
  }
}
