package clearcl.enums;

import coremem.enums.NativeTypeEnum;

/**
 * OpenCL image channel data type
 *
 * @author royer
 */
@SuppressWarnings("javadoc")
public enum ImageChannelDataType
{
 SignedNormalizedInt8(NativeTypeEnum.Byte, true, true),
 SignedNormalizedInt16(NativeTypeEnum.Short, true, true),
 UnsignedNormalizedInt8(NativeTypeEnum.UnsignedByte, true, false),
 UnsignedNormalizedInt16(NativeTypeEnum.UnsignedShort, true, false),
 SignedInt8(NativeTypeEnum.Byte, false, true),
 SignedInt16(NativeTypeEnum.Short, false, true),
 SignedInt32(NativeTypeEnum.Int, false, true),
 UnsignedInt8(NativeTypeEnum.UnsignedByte, false, false),
 UnsignedInt16(NativeTypeEnum.UnsignedShort, false, false),
 UnsignedInt32(NativeTypeEnum.UnsignedInt, false, false),
 HalfFloat(NativeTypeEnum.HalfFloat, false, true),
 Float(NativeTypeEnum.Float, false, true);

  NativeTypeEnum mNativeType;
  boolean mIsNormalized;
  boolean mIsSigned;

  private ImageChannelDataType(NativeTypeEnum pNativeDataType,
                               boolean pIsNormalized,
                               boolean pIsSigned)
  {
    mNativeType = pNativeDataType;
    mIsNormalized = pIsNormalized;
    mIsSigned = pIsSigned;
  }

  /**
   * Return corresponding native type.
   * 
   * @return native type.
   */
  public NativeTypeEnum getNativeType()
  {
    return mNativeType;
  }

  /**
   * Returns true if this image channel data type is normalized
   * 
   * @return true if normalized, false otherwise
   */
  public boolean isNormalized()
  {
    return mIsNormalized;
  }

  /**
   * Returns true if this is a float channel data type.
   * 
   * @return true if float data type, false otherwise
   */
  public boolean isFloat()
  {
    return (this == Float) || (this == HalfFloat);
  }

  /**
   * Returns true if this is an integer channel data type.
   * 
   * @return true if integer data type, false otherwise
   */
  public boolean isInteger()
  {
    return !isFloat();
  }

  /**
   * Returns true if this is channel data type has a sign.
   * 
   * @return true if data type has sign, false otherwise
   */
  public boolean isSigned()
  {
    return mIsSigned;
  }

  /**
   * Returns true if this is channel data type is unsigned.
   * 
   * @return true if data type is unsigned, false otherwise
   */
  public boolean isUnSigned()
  {
    return !mIsSigned;
  }
}
