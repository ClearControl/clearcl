package clearcl.viewer;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import javafx.application.Platform;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleFloatProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.geometry.Insets;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.Border;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.BorderStroke;
import javafx.scene.layout.BorderStrokeStyle;
import javafx.scene.layout.BorderWidths;
import javafx.scene.layout.CornerRadii;
import javafx.scene.paint.Color;

import clearcl.ClearCLBuffer;
import clearcl.ClearCLContext;
import clearcl.ClearCLHostImageBuffer;
import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.ClearCLProgram;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.KernelAccessType;
import clearcl.enums.MemAllocMode;
import clearcl.exceptions.ClearCLUnsupportedException;
import clearcl.interfaces.ClearCLImageInterface;
import clearcl.ocllib.OCLlib;
import clearcl.ops.kernels.CLKernelException;
import clearcl.ops.kernels.CLKernelExecutor;
import clearcl.ops.kernels.Kernels;
import clearcl.util.ElapsedTime;
import clearcl.util.Region2;
import clearcl.viewer.jfx.DirectWritableImage;
import coremem.ContiguousMemoryInterface;
import coremem.enums.NativeTypeEnum;

/**
 * JavaFX Panel for displaying the contents of a ClearCLImage.
 *
 * @author royer
 */
public class ClearCLImagePanel extends BorderPane
{
  private static final float cSmoothingFactor = 0.2f;

  private volatile DirectWritableImage mDirectWritableImage;
  private volatile ImageView mImageView;

  private volatile ClearCLImageInterface mClearCLImage;
  private volatile ClearCLBuffer mRenderRGBBuffer;
  private volatile ClearCLHostImageBuffer mClearCLHostImage;
  private ClearCLProgram mProgramFloat, mProgramUint, mProgramInt;
  private ClearCLKernel mRenderKernel;
  private final CLKernelExecutor mClke;

  private ReentrantLock mLock = new ReentrantLock();

  private final BooleanProperty mIsActive =
                                          new SimpleBooleanProperty(true);
  private final BooleanProperty mAuto =
                                      new SimpleBooleanProperty(true);
  private final FloatProperty mMin = new SimpleFloatProperty(0);
  private final FloatProperty mMax = new SimpleFloatProperty(1);
  private final FloatProperty mGamma = new SimpleFloatProperty(1);
  private final IntegerProperty mZ = new SimpleIntegerProperty(0);
  private final IntegerProperty mNumberOfSteps =
                                               new SimpleIntegerProperty(128);

  private final ObjectProperty<RenderMode> mRenderMode =
                                                       new SimpleObjectProperty<>(RenderMode.Slice);

  private Float mTrueMin = 0f, mTrueMax = 1f;

  /**
   * Creates a panel for a given ClearCL image.
   * 
   * @param pClearCLImage
   *          image
   */
  public ClearCLImagePanel(ClearCLImageInterface pClearCLImage)
  {
    super();
    mClearCLImage = pClearCLImage;

    ClearCLContext lContext = pClearCLImage.getContext();

    if (pClearCLImage.getDimension() == 1)
    {
      throw new ClearCLUnsupportedException("1D image visualizationnot supported");
    }

    ensureBuffersAllocated(mClearCLImage);

    try
    {
      mProgramFloat =
                    lContext.createProgram(OCLlib.class,
                                           "render/img2D.cl",
                                           "render/ortho/avgproj3D.cl",
                                           "render/ortho/colproj3D.cl",
                                           "render/ortho/maxproj3D.cl",
                                           "render/ortho/slice3D.cl");

      mProgramFloat.addBuildOptionAllMathOpt();
      mProgramFloat.addDefine("FLOAT");
      mProgramFloat.buildAndLog();

      mProgramUint =
                   lContext.createProgram(OCLlib.class,
                                          "render/img2D.cl",
                                          "render/ortho/avgproj3D.cl",
                                          "render/ortho/colproj3D.cl",
                                          "render/ortho/maxproj3D.cl",
                                          "render/ortho/slice3D.cl");

      mProgramUint.addBuildOptionAllMathOpt();
      mProgramUint.addDefine("UINT");
      mProgramUint.buildAndLog();

      mProgramInt = lContext.createProgram(OCLlib.class,
                                           "render/img2D.cl",
                                           "render/ortho/avgproj3D.cl",
                                           "render/ortho/colproj3D.cl",
                                           "render/ortho/maxproj3D.cl",
                                           "render/ortho/slice3D.cl");

      mProgramInt.addBuildOptionAllMathOpt();
      mProgramInt.addDefine("INT");
      mProgramInt.buildAndLog();

      if (pClearCLImage.getDimension() == 1)
      {
        throw new ClearCLUnsupportedException("1D image visualizationnot supported");
      }

      mClke = new CLKernelExecutor(lContext, this.getClass());

    }
    catch (IOException e)
    {
      e.printStackTrace();
      throw new RuntimeException("Cannot build program", e);
    }

    backgroundProperty().set(new Background(new BackgroundFill(Color.BLACK,
                                                               CornerRadii.EMPTY,
                                                               Insets.EMPTY)));/**/

    ensureCanvasIsSetup(pClearCLImage);

    addImageListener(pClearCLImage);

    mMin.addListener((e) -> {
      updateImage();
    });
    mMax.addListener((e) -> {
      updateImage();
    });
    mAuto.addListener((e) -> {
      updateImage();
    });
    mGamma.addListener((e) -> {
      updateImage();
    });
    mZ.addListener((e) -> {
      updateImage();
    });
    mRenderMode.addListener((e) -> {
      updateImage();
    });

    mNumberOfSteps.set((int) Math.min(mNumberOfSteps.get(),
                                      pClearCLImage.getDepth()));

    updateImage();
  }

  private void ensureCanvasIsSetup(ClearCLImageInterface pClearCLImage)
  {
    if (mDirectWritableImage != null
        && (int) mDirectWritableImage.getWidth() == (int) pClearCLImage.getWidth()
        && (int) mDirectWritableImage.getHeight() == (int) pClearCLImage.getHeight())
      return;

    if (mDirectWritableImage != null)
      getChildren().remove(mDirectWritableImage);

    mDirectWritableImage =
                         new DirectWritableImage((int) pClearCLImage.getWidth(),
                                                 (int) pClearCLImage.getHeight());

    mImageView = new ImageView(mDirectWritableImage);
    mImageView.setStyle("-fx-padding: 10");
    mImageView.setStyle("-fx-background-color: red");
    mImageView.setStyle("-fx-border-color: red");
    setBorder(new Border(new BorderStroke(Color.RED,
                                          BorderStrokeStyle.SOLID,
                                          CornerRadii.EMPTY,
                                          BorderWidths.DEFAULT)));/**/

    this.setWidth(mDirectWritableImage.getWidth());
    this.setHeight(mDirectWritableImage.getHeight());

    setCenter(mImageView);
    BorderPane.setMargin(mImageView, Insets.EMPTY);
    // StackPane.setAlignment(mImageView, Pos.CENTER);

  }

  private void addImageListener(ClearCLImageInterface pClearCLImage)
  {
    pClearCLImage.addListener((q, s) -> {

      if (getIsActive().get())
      {
        q.waitToFinish();
        updateImage();
      }
    });
  }

  private void ensureBuffersAllocated(ClearCLImageInterface pNewImage)
  {
    if (mRenderRGBBuffer == null
        || mRenderRGBBuffer.getWidth() != pNewImage.getWidth()
        || mRenderRGBBuffer.getHeight() != pNewImage.getHeight())
    {
      if (mRenderRGBBuffer != null)
        mRenderRGBBuffer.close();

      mRenderRGBBuffer =
                       pNewImage.getContext()
                                .createBuffer(MemAllocMode.AllocateHostPointer,
                                              HostAccessType.ReadOnly,
                                              KernelAccessType.WriteOnly,
                                              4,
                                              NativeTypeEnum.Byte,
                                              Region2.region(pNewImage.getDimensions()));
    }

    if (mClearCLHostImage == null
        || mClearCLHostImage.getWidth() != pNewImage.getWidth()
        || mClearCLHostImage.getHeight() != pNewImage.getHeight())
    {
      if (mClearCLHostImage != null)
        mClearCLHostImage.close();
      mClearCLHostImage =
                        ClearCLHostImageBuffer.allocateSameAs(mRenderRGBBuffer);
    }
  }

  /**
   * Sets a new image to be viewed. The image must have the same dimensionality
   * as the original image, but the actual width, height or depth can be
   * different.
   * 
   * @param pImage
   *          new image
   */
  public void setImage(ClearCLImageInterface pImage)
  {
    boolean lTryLock = false;
    try
    {
      lTryLock = mLock.tryLock(1000, TimeUnit.MILLISECONDS);
    }
    catch (InterruptedException e)
    {
    }

    if (lTryLock)
    {
      try
      {
        mClearCLImage.getContext().getDefaultQueue().waitToFinish();
        ensureBuffersAllocated(pImage);
        mClearCLImage = pImage;
        addImageListener(pImage);
      }
      finally
      {
        mLock.unlock();
      }
    }
  }

  /**
   * Updates the display of this ImageView. This is called automatically through
   * an internal listener when the image contents (may) have changed.
   */
  public void updateImage()
  {
    boolean lTryLock = false;
    try
    {
      lTryLock = mLock.tryLock(1, TimeUnit.MILLISECONDS);
    }
    catch (InterruptedException e)
    {
    }

    if (lTryLock)
    {
      try
      {
        // System.out.println("Update View");
        float lMin = 0;
        float lMax = 1;

        if (mAuto.get() || mTrueMin == null)
        {
          try
          {
            float[] lMinMax =
                            Kernels.minMax(mClke, mClearCLImage, 32);
            /*System.out.format("computed: min=%f, max=%f \n",
                            lMinMax[0],
                            lMinMax[1]);/**/

            float lMinValue = lMinMax[0];
            float lMaxValue = lMinMax[1];

            if (Float.isInfinite(lMinValue)
                || Float.isInfinite(lMaxValue))
              System.err.println("Image has infinite value! "
                                 + mClearCLImage);
            else
            {

              mTrueMin = (1 - cSmoothingFactor) * lMinValue
                         + cSmoothingFactor * mTrueMin;
              mTrueMax = (1 - cSmoothingFactor) * lMaxValue
                         + cSmoothingFactor * mTrueMax;

            }

            lMin = mTrueMin;
            lMax = mTrueMax;
          }
          catch (CLKernelException clke)
          {
            lMin = mTrueMin + (mTrueMax - mTrueMin) * mMin.get();
            lMax = mTrueMin + (mTrueMax - mTrueMin) * mMax.get();
          }

        }
        else
        {
          lMin = mTrueMin + (mTrueMax - mTrueMin) * mMin.get();
          lMax = mTrueMin + (mTrueMax - mTrueMin) * mMax.get();
        }

        /*System.out.format("true:    min=%f, max=%f \n",
                          mTrueMin,
                          mTrueMax);
        System.out.format("current: min=%f, max=%f \n", lMin, lMax);/**/

        if (mClearCLImage.getDimension() == 2)
        {
          if (mClearCLImage instanceof ClearCLImage)
          {
            ClearCLImage lImage = (ClearCLImage) mClearCLImage;
            ImageChannelDataType lDataType =
                                           lImage.getChannelDataType();

            if (lDataType.isNormalized() || lDataType.isFloat())
              mRenderKernel =
                            mProgramFloat.getKernel("image_render_2d");
            else if (lDataType.isInteger() && lDataType.isUnSigned())
              mRenderKernel =
                            mProgramUint.getKernel("image_render_2d");
            else if (lDataType.isInteger() && lDataType.isSigned())
              mRenderKernel =
                            mProgramInt.getKernel("image_render_2d");

          }
          else if (mClearCLImage instanceof ClearCLBuffer)
            mRenderKernel =
                          mProgramFloat.getKernel("buffer_render_2df");

        }
        else if (mClearCLImage.getDimension() == 3)
        {
          switch (getRenderModeProperty().get())
          {
          case AvgProjection:
            if (mClearCLImage instanceof ClearCLImage)
            {
              ClearCLImage lImage = (ClearCLImage) mClearCLImage;
              ImageChannelDataType lDataType =
                                             lImage.getChannelDataType();

              if (lDataType.isNormalized() || lDataType.isFloat())
                mRenderKernel =
                              mProgramFloat.getKernel("image_render_avgproj_3d");
              else if (lDataType.isInteger()
                       && lDataType.isUnSigned())
                mRenderKernel =
                              mProgramUint.getKernel("image_render_avgproj_3d");
              else if (lDataType.isInteger() && lDataType.isSigned())
                mRenderKernel =
                              mProgramInt.getKernel("image_render_avgproj_3d");

            }
            else if (mClearCLImage instanceof ClearCLBuffer)
              mRenderKernel =
                            mProgramFloat.getKernel("buffer_render_avgproj_3df");
            break;
          case ColorProjection:
            if (mClearCLImage instanceof ClearCLImage)
            {
              ClearCLImage lImage = (ClearCLImage) mClearCLImage;
              ImageChannelDataType lDataType =
                                             lImage.getChannelDataType();

              if (lDataType.isNormalized() || lDataType.isFloat())
                mRenderKernel =
                              mProgramFloat.getKernel("image_render_colorproj_3d");
              else if (lDataType.isInteger()
                       && lDataType.isUnSigned())
                mRenderKernel =
                              mProgramUint.getKernel("image_render_colorproj_3d");
              else if (lDataType.isInteger() && lDataType.isSigned())
                mRenderKernel =
                              mProgramInt.getKernel("image_render_colorproj_3d");

            }
            else if (mClearCLImage instanceof ClearCLBuffer)
              mRenderKernel =
                            mProgramFloat.getKernel("buffer_render_colorproj_3df");
            break;
          case MaxProjection:
            if (mClearCLImage instanceof ClearCLImage)
            {
              ClearCLImage lImage = (ClearCLImage) mClearCLImage;
              ImageChannelDataType lDataType =
                                             lImage.getChannelDataType();

              if (lDataType.isNormalized() || lDataType.isFloat())
                mRenderKernel =
                              mProgramFloat.getKernel("image_render_maxproj_3d");
              else if (lDataType.isInteger()
                       && lDataType.isUnSigned())
                mRenderKernel =
                              mProgramUint.getKernel("image_render_maxproj_3d");
              else if (lDataType.isInteger() && lDataType.isSigned())
                mRenderKernel =
                              mProgramInt.getKernel("image_render_maxproj_3d");

            }
            else if (mClearCLImage instanceof ClearCLBuffer)
              mRenderKernel =
                            mProgramFloat.getKernel("buffer_render_maxproj_3df");
            break;
          default:
          case Slice:
            if (mClearCLImage instanceof ClearCLImage)
            {
              ClearCLImage lImage = (ClearCLImage) mClearCLImage;
              ImageChannelDataType lDataType =
                                             lImage.getChannelDataType();

              if (lDataType.isNormalized() || lDataType.isFloat())
                mRenderKernel =
                              mProgramFloat.getKernel("image_render_slice_3d");
              else if (lDataType.isInteger()
                       && lDataType.isUnSigned())
                mRenderKernel =
                              mProgramUint.getKernel("image_render_slice_3d");
              else if (lDataType.isInteger() && lDataType.isSigned())
                mRenderKernel =
                              mProgramInt.getKernel("image_render_slice_3d");

            }
            else if (mClearCLImage instanceof ClearCLBuffer)
              mRenderKernel =
                            mProgramFloat.getKernel("buffer_render_slice_3df");
            break;
          }

        }

        mRenderKernel.setGlobalSizes(Region2.region(mClearCLImage.getDimensions()));

        mRenderKernel.setArgument("image", mClearCLImage);
        mRenderKernel.setArgument("rgbabuffer", mRenderRGBBuffer);

        mRenderKernel.setArgument("vmin", lMin);
        mRenderKernel.setArgument("vmax", lMax);
        mRenderKernel.setArgument("gamma", mGamma.get());
        mRenderKernel.setOptionalArgument("z", mZ.get());

        final int lZStep = (int) (1.0 * mClearCLImage.getDepth()
                                  / mNumberOfSteps.get());
        mRenderKernel.setOptionalArgument("zstep", lZStep);

        mRenderKernel.run(true);

        ElapsedTime.measure("mRenderRGBBuffer.copyTo(mClearCLHostImage, true);",
                            () -> mRenderRGBBuffer.copyTo(mClearCLHostImage,
                                                          true));

        long lWidth = mClearCLHostImage.getWidth();
        long lHeight = mClearCLHostImage.getHeight();

        ContiguousMemoryInterface lContiguousMemory =
                                                    mClearCLHostImage.getContiguousMemory();

        Platform.runLater(() -> {

          boolean lTryLock2 = false;
          try
          {
            lTryLock2 = mLock.tryLock(1, TimeUnit.MILLISECONDS);
          }
          catch (InterruptedException e)
          {
          }

          if (lTryLock2)
          {
            try
            {
              if (mClearCLImage.getWidth() != lWidth
                  || mClearCLImage.getHeight() != lHeight)
                return;

              ensureCanvasIsSetup(mClearCLImage);
              mDirectWritableImage.replaceBuffer(lContiguousMemory);

            }
            finally
            {
              mLock.unlock();
            }
          }
        });
      }
      finally
      {
        mLock.unlock();
      }
    }
  }

  /**
   * Returns the is-active property that decides whether to react to image
   * updates or not.
   * 
   * @return is-active property
   */
  public BooleanProperty getIsActive()
  {
    return mIsActive;
  }

  /**
   * Returns auto property
   * 
   * @return auto property
   */
  public BooleanProperty getAutoProperty()
  {
    return mAuto;
  }

  /**
   * Returns min property
   * 
   * @return min property
   */
  public FloatProperty getMinProperty()
  {
    return mMin;
  }

  /**
   * Returns max property
   * 
   * @return max property
   */
  public FloatProperty getMaxProperty()
  {
    return mMax;
  }

  /**
   * Returns gamma property
   * 
   * @return gamma property
   */
  public FloatProperty getGammaProperty()
  {
    return mGamma;
  }

  /**
   * Returns z property
   * 
   * @return z property
   */
  public IntegerProperty getZProperty()
  {
    return mZ;
  }

  /**
   * Returns render mode property
   * 
   * @return render mode property
   */
  public ObjectProperty<RenderMode> getRenderModeProperty()
  {
    return mRenderMode;
  }

}
