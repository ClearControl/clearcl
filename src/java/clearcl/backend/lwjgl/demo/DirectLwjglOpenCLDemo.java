/*
 * Copyright LWJGL. All rights reserved.
 * License terms: http://lwjgl.org/license.php
 */
package clearcl.backend.lwjgl.demo;

import static org.lwjgl.opencl.CL10.CL_COMPLETE;
import static org.lwjgl.opencl.CL10.CL_CONTEXT_PLATFORM;
import static org.lwjgl.opencl.CL10.CL_DEVICE_ADDRESS_BITS;
import static org.lwjgl.opencl.CL10.CL_DEVICE_AVAILABLE;
import static org.lwjgl.opencl.CL10.CL_DEVICE_COMPILER_AVAILABLE;
import static org.lwjgl.opencl.CL10.CL_DEVICE_EXECUTION_CAPABILITIES;
import static org.lwjgl.opencl.CL10.CL_DEVICE_EXTENSIONS;
import static org.lwjgl.opencl.CL10.CL_DEVICE_MAX_CLOCK_FREQUENCY;
import static org.lwjgl.opencl.CL10.CL_DEVICE_MAX_COMPUTE_UNITS;
import static org.lwjgl.opencl.CL10.CL_DEVICE_MAX_WORK_GROUP_SIZE;
import static org.lwjgl.opencl.CL10.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
import static org.lwjgl.opencl.CL10.CL_DEVICE_NAME;
import static org.lwjgl.opencl.CL10.CL_DEVICE_PROFILE;
import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE;
import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_ALL;
import static org.lwjgl.opencl.CL10.CL_DEVICE_VENDOR;
import static org.lwjgl.opencl.CL10.CL_DEVICE_VENDOR_ID;
import static org.lwjgl.opencl.CL10.CL_DEVICE_VERSION;
import static org.lwjgl.opencl.CL10.CL_DRIVER_VERSION;
import static org.lwjgl.opencl.CL10.CL_EXEC_NATIVE_KERNEL;
import static org.lwjgl.opencl.CL10.CL_MEM_READ_ONLY;
import static org.lwjgl.opencl.CL10.CL_PLATFORM_EXTENSIONS;
import static org.lwjgl.opencl.CL10.CL_PLATFORM_NAME;
import static org.lwjgl.opencl.CL10.CL_PLATFORM_PROFILE;
import static org.lwjgl.opencl.CL10.CL_PLATFORM_VENDOR;
import static org.lwjgl.opencl.CL10.CL_PLATFORM_VERSION;
import static org.lwjgl.opencl.CL10.CL_QUEUED;
import static org.lwjgl.opencl.CL10.CL_RUNNING;
import static org.lwjgl.opencl.CL10.CL_SUBMITTED;
import static org.lwjgl.opencl.CL10.clCreateBuffer;
import static org.lwjgl.opencl.CL10.clCreateCommandQueue;
import static org.lwjgl.opencl.CL10.clCreateContext;
import static org.lwjgl.opencl.CL10.clEnqueueNativeKernel;
import static org.lwjgl.opencl.CL10.clFinish;
import static org.lwjgl.opencl.CL10.clReleaseCommandQueue;
import static org.lwjgl.opencl.CL10.clReleaseContext;
import static org.lwjgl.opencl.CL10.clReleaseEvent;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;
import static org.lwjgl.opencl.CL11.CL_BUFFER_CREATE_TYPE_REGION;
import static org.lwjgl.opencl.CL11.CL_DEVICE_OPENCL_C_VERSION;
import static org.lwjgl.opencl.CL11.clSetEventCallback;
import static org.lwjgl.opencl.CL11.clSetMemObjectDestructorCallback;
import static org.lwjgl.opencl.CL11.nclCreateSubBuffer;
import static org.lwjgl.opencl.CLUtil.checkCLError;
import static org.lwjgl.opencl.Info.clGetDeviceInfoBoolean;
import static org.lwjgl.opencl.Info.clGetDeviceInfoInt;
import static org.lwjgl.opencl.Info.clGetDeviceInfoLong;
import static org.lwjgl.opencl.Info.clGetDeviceInfoPointer;
import static org.lwjgl.opencl.Info.clGetDeviceInfoStringUTF8;
import static org.lwjgl.opencl.Info.clGetPlatformInfoStringUTF8;
import static org.lwjgl.opencl.KHRICD.CL_PLATFORM_ICD_SUFFIX_KHR;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memByteBuffer;
import static org.lwjgl.system.MemoryUtil.memDecodeUTF8;
import static org.lwjgl.system.Pointer.POINTER_SIZE;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLBufferRegion;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLEventCallback;
import org.lwjgl.opencl.CLMemObjectDestructorCallback;
import org.lwjgl.opencl.CLNativeKernel;
import org.lwjgl.opencl.CLPlatform;
import org.lwjgl.system.FunctionProviderLocal;

public final class DirectLwjglOpenCLDemo {

	private DirectLwjglOpenCLDemo() {
	}

	public static void main(String[] args) {
		System.setProperty(
			"org.lwjgl.system.libffi.ClosureRegistry",
			"clearcl.lwjgl.demo.ClosureGC"
		);

		List<CLPlatform> platforms = CLPlatform.getPlatforms();
		if ( platforms.isEmpty() )
			throw new RuntimeException("No OpenCL platforms found.");

		IntBuffer errcode_ret = BufferUtils.createIntBuffer(1);

		for ( CLPlatform platform : platforms ) {
			System.out.println("\n-------------------------");
			System.out.printf("NEW PLATFORM: [0x%X]\n", platform.address());

			CLCapabilities platformCaps = platform.getCapabilities();

			printPlatformInfo(platform, "CL_PLATFORM_PROFILE", CL_PLATFORM_PROFILE);
			printPlatformInfo(platform, "CL_PLATFORM_VERSION", CL_PLATFORM_VERSION);
			printPlatformInfo(platform, "CL_PLATFORM_NAME", CL_PLATFORM_NAME);
			printPlatformInfo(platform, "CL_PLATFORM_VENDOR", CL_PLATFORM_VENDOR);
			printPlatformInfo(platform, "CL_PLATFORM_EXTENSIONS", CL_PLATFORM_EXTENSIONS);
			if ( platformCaps.cl_khr_icd )
				printPlatformInfo(platform, "CL_PLATFORM_ICD_SUFFIX_KHR", CL_PLATFORM_ICD_SUFFIX_KHR);
			System.out.println("");

			PointerBuffer ctxProps = BufferUtils.createPointerBuffer(3);
			ctxProps.put(CL_CONTEXT_PLATFORM).put(platform).put(0).flip();

			List<CLDevice> devices = platform.getDevices(CL_DEVICE_TYPE_ALL);
			for ( CLDevice device : devices ) {
				long cl_device_id = device.address();
				CLCapabilities caps = device.getCapabilities();

				System.out.printf("\n\t** NEW DEVICE: [0x%X]\n", cl_device_id);
				System.out.print("\t");
				System.out.println(caps);

				System.out.println("\tCL_DEVICE_TYPE = " + clGetDeviceInfoLong(cl_device_id, CL_DEVICE_TYPE));
				System.out.println("\tCL_DEVICE_VENDOR_ID = " + clGetDeviceInfoInt(cl_device_id, CL_DEVICE_VENDOR_ID));
				System.out.println("\tCL_DEVICE_MAX_COMPUTE_UNITS = " + clGetDeviceInfoInt(cl_device_id, CL_DEVICE_MAX_COMPUTE_UNITS));
				System.out
					.println("\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = " + clGetDeviceInfoInt(cl_device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
				System.out.println("\tCL_DEVICE_MAX_WORK_GROUP_SIZE = " + clGetDeviceInfoPointer(cl_device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE));
				System.out.println("\tCL_DEVICE_MAX_CLOCK_FREQUENCY = " + clGetDeviceInfoInt(cl_device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY));
				System.out.println("\tCL_DEVICE_ADDRESS_BITS = " + clGetDeviceInfoInt(cl_device_id, CL_DEVICE_ADDRESS_BITS));
				System.out.println("\tCL_DEVICE_AVAILABLE = " + clGetDeviceInfoBoolean(cl_device_id, CL_DEVICE_AVAILABLE));
				System.out.println("\tCL_DEVICE_COMPILER_AVAILABLE = " + clGetDeviceInfoBoolean(cl_device_id, CL_DEVICE_COMPILER_AVAILABLE));

				printDeviceInfo(device, "CL_DEVICE_NAME", CL_DEVICE_NAME);
				printDeviceInfo(device, "CL_DEVICE_VENDOR", CL_DEVICE_VENDOR);
				printDeviceInfo(device, "CL_DRIVER_VERSION", CL_DRIVER_VERSION);
				printDeviceInfo(device, "CL_DEVICE_PROFILE", CL_DEVICE_PROFILE);
				printDeviceInfo(device, "CL_DEVICE_VERSION", CL_DEVICE_VERSION);
				printDeviceInfo(device, "CL_DEVICE_EXTENSIONS", CL_DEVICE_EXTENSIONS);
				if ( caps.OpenCL11 )
					printDeviceInfo(device, "CL_DEVICE_OPENCL_C_VERSION", CL_DEVICE_OPENCL_C_VERSION);

				long context = clCreateContext(ctxProps, device.address(), new CLContextCallback() {
					@Override
					public void invoke(long errinfo, long private_info, long cb, long user_data) {
						System.err.println("[LWJGL] cl_context_callback");
						System.err.println("\tInfo: " + memDecodeUTF8(errinfo));
					}
				}, NULL, errcode_ret);
				checkCLError(errcode_ret);

				long buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 128, errcode_ret);
				checkCLError(errcode_ret);

				int errcode;

				if ( caps.OpenCL11 ) {
					errcode = clSetMemObjectDestructorCallback(buffer, new CLMemObjectDestructorCallback() {
						@Override
						public void invoke(long memobj, long user_data) {
							System.out.println("\t\tBuffer destructed (1): " + memobj);
						}
					}, NULL);
					checkCLError(errcode);

					errcode = clSetMemObjectDestructorCallback(buffer, new CLMemObjectDestructorCallback() {
						@Override
						public void invoke(long memobj, long user_data) {
							System.out.println("\t\tBuffer destructed (2): " + memobj);
						}
					}, NULL);
					checkCLError(errcode);
				}

				long subbuffer = NULL;
				if ( caps.OpenCL11 ) {
					CLBufferRegion buffer_region = CLBufferRegion.malloc();
					buffer_region.origin(0);
					buffer_region.size(64);

					try {
						subbuffer = nclCreateSubBuffer(buffer, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, buffer_region.address(), memAddress(errcode_ret));
						checkCLError(errcode_ret);
					} finally {
						buffer_region.free();
					}

					errcode = clSetMemObjectDestructorCallback(subbuffer, new CLMemObjectDestructorCallback() {
						@Override
						public void invoke(long memobj, long user_data) {
							System.out.println("\t\tSub Buffer destructed: " + memobj);
						}
					}, NULL);
					checkCLError(errcode);
				}

				long exec_caps = clGetDeviceInfoLong(cl_device_id, CL_DEVICE_EXECUTION_CAPABILITIES);
				if ( (exec_caps & CL_EXEC_NATIVE_KERNEL) == CL_EXEC_NATIVE_KERNEL ) {
					ClosureGC.get().push();

					System.out.println("\t\t-TRYING TO EXEC NATIVE KERNEL-");
					long queue = clCreateCommandQueue(context, device.address(), 0L, errcode_ret);

					PointerBuffer ev = BufferUtils.createPointerBuffer(1);

					ByteBuffer kernelArgs = BufferUtils.createByteBuffer(4);
					kernelArgs.putInt(0, 1337);

					errcode = clEnqueueNativeKernel(queue, new CLNativeKernel() {
						@Override
						public void invoke(long args) {
							System.out.println("\t\tKERNEL EXEC argument: " + memByteBuffer(args, 4).getInt(0) + ", should be 1337");
						}
					}, kernelArgs, null, null, null, ev);
					checkCLError(errcode);

					long e = ev.get(0);

					final CountDownLatch latch = new CountDownLatch(1);
					errcode = clSetEventCallback(e, CL_COMPLETE, new CLEventCallback() {
						@Override
						public void invoke(long event, int event_command_exec_status, long user_data) {
							System.out.println("\t\tEvent callback status: " + getEventStatusName(event_command_exec_status));
							latch.countDown();
						}
					}, NULL);
					checkCLError(errcode);

					try {
						boolean expired = !latch.await(500, TimeUnit.MILLISECONDS);
						if ( expired )
							System.out.println("\t\tKERNEL EXEC FAILED!");
					} catch (InterruptedException exc) {
						exc.printStackTrace();
					}

					errcode = clReleaseEvent(e);
					checkCLError(errcode);

					kernelArgs = BufferUtils.createByteBuffer(POINTER_SIZE * 2);

					CLNativeKernel kernel = new CLNativeKernel() {
						@Override
						public void invoke(long args) {
						}
					};

					long time = System.nanoTime();
					int REPEAT = 1000;
					for ( int i = 0; i < REPEAT; i++ )
						clEnqueueNativeKernel(queue, kernel, kernelArgs, null, null, null, null);
					clFinish(queue);
					time = System.nanoTime() - time;

					System.out.printf("\n\t\tEMPTY NATIVE KERNEL AVG EXEC TIME: %.4fus\n", (double)time / (REPEAT * 1000));

					errcode = clReleaseCommandQueue(queue);
					checkCLError(errcode);

					ClosureGC.get().pop();
				}

				System.out.println();

				if ( subbuffer != NULL ) {
					errcode = clReleaseMemObject(subbuffer);
					checkCLError(errcode);
				}

				errcode = clReleaseMemObject(buffer);
				checkCLError(errcode);

				errcode = clReleaseContext(context);
				checkCLError(errcode);

				ClosureGC.get().gc();
			}
		}
	}

	public static void get(FunctionProviderLocal provider, long platform, String name) {
		System.out.println(name + ": " + provider.getFunctionAddress(platform, name));
	}

	private static void printPlatformInfo(CLPlatform platform, String param_name, int param) {
		System.out.println("\t" + param_name + " = " + clGetPlatformInfoStringUTF8(platform.address(), param));
	}

	private static void printDeviceInfo(CLDevice device, String param_name, int param) {
		System.out.println("\t" + param_name + " = " + clGetDeviceInfoStringUTF8(device.address(), param));
	}

	private static String getEventStatusName(int status) {
		switch ( status ) {
			case CL_QUEUED:
				return "CL_QUEUED";
			case CL_SUBMITTED:
				return "CL_SUBMITTED";
			case CL_RUNNING:
				return "CL_RUNNING";
			case CL_COMPLETE:
				return "CL_COMPLETE";
			default:
				throw new IllegalArgumentException(String.format("Unsupported event status: 0x%X", status));
		}
	}

}